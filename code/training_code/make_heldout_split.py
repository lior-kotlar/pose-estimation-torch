"""Freeze one train / val / test split of the labelled frames, shared by every
ensemble member.

Why a file and not a split drawn per run: the ensemble's accuracy can only be
measured on frames NO member trained on, and members can only be compared on
the same frames. A per-run shuffle gives neither -- the three 3-camera members
trained on Aug 20 do not share a single validation frame.

Why clumps and not single frames: the labelled set holds pairs of frames from
the same flight a few frames apart (body within a few degrees, crops within a
few pixels on every camera, wings elsewhere). Splitting frame by frame leaves
about half the held-out frames with such a twin in training. Frames are
therefore joined into clumps -- two frames are neighbours when their crops
are within CROP_PX on all cameras AND their body axes within AXIS_DEG, and
neighbours of neighbours share a clump -- and whole clumps are assigned.

    test  -- never trained on, never used to pick a checkpoint
    val   -- picks each member's checkpoint
    train -- the rest

Usage:
    python code/training_code/make_heldout_split.py <dataset.h5> [--out PATH]
The output defaults to <dataset>.split_v1.npz next to the dataset, and the
script refuses to overwrite an existing split: every member must see the same
one, so a new split is a new version, not an edit.
"""
import argparse
import hashlib
import json
import os

import h5py
import numpy as np

CROP_PX = 20
AXIS_DEG = 10.0
TEST_FRACTION = 0.2
VAL_FRACTION = 0.1
SEED = 20260927

TRAIN, VAL, TEST = 0, 1, 2


def md5_of(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def find_clumps(crop_zone, points_3d):
    """crop_zone (N, cams, 2), points_3d (N, joints, 3) with head, tail last.
    Returns a clump id per frame (connected components of the neighbour graph)."""
    crop = crop_zone.astype(float)
    crop_dist = np.abs(crop[:, None] - crop[None]).max(axis=(2, 3))
    axis = points_3d[:, -2] - points_3d[:, -1]
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    axis_deg = np.degrees(np.arccos(np.clip(axis @ axis.T, -1, 1)))
    neighbours = (crop_dist < CROP_PX) & (axis_deg < AXIS_DEG)
    np.fill_diagonal(neighbours, False)

    n = len(crop)
    clump = -np.ones(n, dtype=int)
    next_id = 0
    for start in range(n):
        if clump[start] >= 0:
            continue
        clump[start] = next_id
        stack = [start]
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(neighbours[u] & (clump < 0)):
                clump[v] = next_id
                stack.append(v)
        next_id += 1
    return clump, crop_dist


def take_clumps(order, clump, available, n_wanted):
    """Walk clumps in `order`, taking whole ones from `available` frames
    until at least n_wanted frames are taken."""
    taken = np.zeros(len(clump), dtype=bool)
    for c in order:
        if taken.sum() >= n_wanted:
            break
        members = (clump == c) & available
        if members.any() and members.sum() == (clump == c).sum():
            taken |= members
    return taken


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = args.out or os.path.splitext(args.dataset)[0] + ".split_v1.npz"
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; a split is never rewritten -- use a new --out")

    with h5py.File(args.dataset, "r") as f:
        crop_zone = f["cropZone"][:]                          # (N, cams, 2)
        points_3d = f["points_3D"][:].transpose(1, 2, 0)      # (N, joints, 3)
    n = len(crop_zone)

    clump, crop_dist = find_clumps(crop_zone, points_3d)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(clump.max() + 1)

    is_test = take_clumps(order, clump, np.ones(n, bool), int(round(n * TEST_FRACTION)))
    rest = [c for c in order if not is_test[clump == c].any()]
    is_val = take_clumps(rest, clump, ~is_test, int(round(n * VAL_FRACTION)))

    frame_split = np.full(n, TRAIN, dtype=np.int8)
    frame_split[is_val] = VAL
    frame_split[is_test] = TEST

    # Sanity: no held-out frame may have a neighbour on another side.
    for name, side in (("test", TEST), ("val", VAL)):
        held = frame_split == side
        closest_other = crop_dist[np.ix_(held, ~held)].min()
        print(f"{name}: {held.sum()} frames in {len(np.unique(clump[held]))} clumps; "
              f"closest frame outside it: {closest_other:.0f} px")
    print(f"train: {(frame_split == TRAIN).sum()} frames")

    meta = {
        "dataset": os.path.abspath(args.dataset),
        "dataset_md5": md5_of(args.dataset),
        "n_frames": n,
        "crop_px": CROP_PX, "axis_deg": AXIS_DEG,
        "test_fraction": TEST_FRACTION, "val_fraction": VAL_FRACTION,
        "seed": SEED,
        "labels": {"train": TRAIN, "val": VAL, "test": TEST},
    }
    np.savez(out, frame_split=frame_split, clump_id=clump, meta=json.dumps(meta))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

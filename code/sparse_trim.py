"""Blank the frames of a sparse mat that prep will never read, so a movie travels to the cluster
at a fraction of its size and still preps exactly as the original would.

Prep reads every frame of a movie's `*_sparse.mat` files in only two places it cares about the
result of: the prescan, which picks the longest run of frames where the fly is tracked, and the
MATLAB build, which then extracts that run plus `MATLAB_TIME_JUMP_MARGIN` frames of padding on
each side (scan_sparse_movies.build_range / build_read_range). The mirror check reads a fixed,
evenly spaced sample of frames (find_mirror_cam.sample_frame_indices). Everything else -- often
half the bytes or more -- only ever reaches the raw movie.

So a PC that runs the prescan itself can keep, per camera:

    the build's read range  +  the mirror check's sample frames

and replace every other frame with MATLAB's own empty placeholder. Frame count, `startFrame` and
every kept frame are untouched, so:

  - the cluster's prescan finds the same run: blanking can only make a frame "not good", the run
    is bounded on both sides by frames that were already not good (and are kept, as padding),
    and no kept sample frame outside it can join or outgrow it;
  - the build reads identical frames and writes an identical box h5, under the same name;
  - the mirror check reads identical pixels and reaches an identical verdict;
  - trigger-relative numbering (startFrame + start_ind - 1 + k) is unchanged.

The one product that differs is the raw movie, which shows only the kept frames.

A blanked movie carries `trim.json` beside its mats, recording what was kept. Prep reads it
(process_experiment.py) and refuses a movie whose build would reach a blanked frame, so a
disagreement between the PC's prescan and the cluster's can never build from missing data.

The file is written the way MATLAB writes a v7.3 mat: a 512-byte MATLAB header in the HDF5
userblock, `/frames/indIm` a (1, N) array of object references into `/#refs#`, an empty frame a
(2,) uint64 dataset holding its dimensions with `MATLAB_empty` = 1. References do not survive a
copy from one file to another, so every reference dataset is rewritten to point at the copies.

Usage, on one movie (normally driven by code/predict_prep.py):

    .env/bin/python code/sparse_trim.py <movie_dir> <out_dir>
"""
import datetime as dt
import glob
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from find_mirror_cam import PREP_SAMPLES, sample_frame_indices  # noqa: E402
from scan_sparse_movies import (PRESCAN_DEFAULTS, build_range,  # noqa: E402
                                build_read_range, scan_movie)

TRIM_FILE = "trim.json"
SCHEMA = 1
USERBLOCK = 512
FRAMES = "frames/indIm"
REFS = "#refs#"


# ---------------------------------------------------------------------------
# Which frames to keep
# ---------------------------------------------------------------------------
def merge_ranges(indices) -> list:
    """Sorted 0-based frame indices -> [[first, last], ...], both ends inclusive."""
    ranges = []
    for i in sorted(set(int(i) for i in indices)):
        if ranges and i == ranges[-1][1] + 1:
            ranges[-1][1] = i
        else:
            ranges.append([i, i])
    return ranges


def in_ranges(i: int, ranges: list) -> bool:
    return any(a <= i <= b for a, b in ranges)


def frames_to_keep(n_frames: int, start_ind: int, end_ind: int,
                   n_samples: int = PREP_SAMPLES) -> list:
    """The 0-based frames of one camera's mat that prep can read: the build's read range and the
    mirror check's sample. Clipped to the mat, whose length may differ between cameras."""
    first, last = build_read_range(start_ind, end_ind)
    keep = set(range(max(first, 0), min(last, n_frames - 1) + 1))
    keep.update(sample_frame_indices(n_frames, n_samples))
    return sorted(keep)


# ---------------------------------------------------------------------------
# Writing a blanked mat
# ---------------------------------------------------------------------------
def _is_ref_dataset(obj) -> bool:
    return isinstance(obj, h5py.Dataset) and h5py.check_ref_dtype(obj.dtype) is h5py.Reference


def _names_by_address(f) -> dict:
    """{object address: path} for everything in `#refs#`. h5py's `.name` on a dereferenced object
    searches the whole file for a path (~6 ms a call), which for thousands of frames is minutes;
    an object's address is known the moment it is opened."""
    group = f[REFS]
    return {h5py.h5o.get_info(group[n].id).addr: f"/{REFS}/{n}" for n in group}


def _ref_name(f, ref, by_address: dict) -> str:
    addr = h5py.h5o.get_info(h5py.h5r.dereference(ref, f.id)).addr
    return by_address.get(addr) or f[ref].name


def _resolves(f, ref) -> bool:
    try:
        return f[ref] is not None
    except (ValueError, KeyError, RuntimeError):
        return False


def _empty_template(src, refs):
    """The value and dtype MATLAB uses for an empty frame in this file, taken from one of its own
    empty frames when it has one."""
    for r in refs:
        d = src[r]
        if d.attrs.get("MATLAB_empty", 0):
            return d[()], d.dtype
    return np.array([0, 3], dtype=np.uint64), np.dtype(np.uint64)


def blank_mat(src_path: str, dst_path: str, keep) -> dict:
    """Write a copy of the sparse mat `src_path` to `dst_path` with every frame not in `keep`
    replaced by an empty frame. Returns counts. The copy lands under a temporary name and is
    moved into place only once it is complete.

    Every frame left out points at ONE empty frame, of the class and shape MATLAB gives its own
    empty frames: a frame is a reference, and an object per empty frame would keep a few MB of
    HDF5 bookkeeping in every mat for nothing."""
    keep = set(int(i) for i in keep)
    staged = dst_path + ".partial"
    if os.path.exists(staged):
        os.remove(staged)
    with h5py.File(src_path, "r") as src:
        refs = src[FRAMES][0]
        n = len(refs)
        by_address = _names_by_address(src)
        frame_names = [_ref_name(src, r, by_address) for r in refs]
        frame_set = set(frame_names)
        empty_value, empty_dtype = _empty_template(src, refs)
        kept = blanked = 0
        with h5py.File(staged, "w", userblock_size=USERBLOCK, libver="earliest") as dst:
            for k, v in src.attrs.items():
                dst.attrs[k] = v
            # every object that is not a frame, as it is: the canonical empty, the camera
            # header's cells (metaData/xmlStruct), metaData itself, anything else MATLAB put here
            dst_refs = dst.create_group(REFS)
            for k, v in src[REFS].attrs.items():
                dst_refs.attrs[k] = v
            for name, obj in src[REFS].items():
                if f"/{REFS}/{name}" not in frame_set:
                    src.copy(obj, dst_refs, name=name)
            for name, obj in src.items():
                if name not in (REFS, "frames"):
                    src.copy(obj, dst, name=name)
            # the frames: kept ones copied as they are (compression and attributes included),
            # the rest pointed at the shared empty
            targets, shared = [], None
            for i, full in enumerate(frame_names):
                if i in keep:
                    name = full.rsplit("/", 1)[1]
                    if name not in dst_refs:   # two frames may share one object
                        src.copy(src[full], dst_refs, name=name)
                    targets.append(full)
                    kept += 1
                    continue
                if shared is None:
                    obj = src[full]
                    shared = full
                    d = dst_refs.create_dataset(full.rsplit("/", 1)[1], data=empty_value,
                                                dtype=empty_dtype)
                    d.attrs["H5PATH"] = np.bytes_(full.encode())
                    d.attrs["MATLAB_class"] = obj.attrs.get("MATLAB_class", np.bytes_(b"uint16"))
                    d.attrs["MATLAB_empty"] = np.uint8(1)
                targets.append(shared)
                blanked += 1
            frame_names = targets
            frames = dst.create_group("frames")
            for k, v in src["frames"].attrs.items():
                frames.attrs[k] = v
            src_ind = src[FRAMES]
            ind = frames.create_dataset(
                "indIm", shape=src_ind.shape, dtype=h5py.ref_dtype,
                chunks=src_ind.chunks, compression=src_ind.compression,
                compression_opts=src_ind.compression_opts)
            for k, v in src_ind.attrs.items():
                ind.attrs[k] = v
            ind[0, :] = np.array([dst[name].ref for name in frame_names], dtype=h5py.ref_dtype)
            # every other reference was copied pointing into the SOURCE file: re-point it
            pending = []
            dst.visititems(lambda path, obj: pending.append(path)
                           if _is_ref_dataset(obj) and path != FRAMES else None)
            for path in pending:
                old = src[path][()]
                new = np.empty(old.shape, dtype=h5py.ref_dtype)
                for idx, r in np.ndenumerate(old):
                    new[idx] = dst[_ref_name(src, r, by_address)].ref if r else r
                dst[path][...] = new
    with open(src_path, "rb") as f:
        header = f.read(USERBLOCK)
    with open(staged, "r+b") as f:
        f.write(header)
    os.replace(staged, dst_path)
    return {"n_frames": n, "kept_frames": kept, "blanked_frames": blanked}


def check_blanked(src_path: str, dst_path: str, keep) -> "str | None":
    """None when `dst_path` is `src_path` with exactly the frames outside `keep` emptied, else
    what is wrong. Reads every kept frame of both files."""
    keep = set(int(i) for i in keep)
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "r") as dst:
        if dst.userblock_size != USERBLOCK:
            return "no MATLAB header block"
        a, b = src[FRAMES][0], dst[FRAMES][0]
        if len(a) != len(b):
            return f"{len(b)} frames instead of {len(a)}"
        for key in ("bg", "frameRate", "frameSize", "startFrame"):
            if not np.array_equal(src["metaData"][key][()], dst["metaData"][key][()]):
                return f"metaData.{key} differs"
        for i in range(len(a)):
            s, d = src[a[i]], dst[b[i]]
            if i in keep:
                if s.shape != d.shape or not np.array_equal(s[()], d[()]):
                    return f"kept frame {i} differs"
            elif not d.attrs.get("MATLAB_empty", 0):
                return f"frame {i} should be empty"
        paths = []
        dst.visititems(lambda path, obj: paths.append(path)
                       if _is_ref_dataset(obj) and path != FRAMES else None)
        for path in paths:
            for r in dst[path][()].flat:
                if r and not _resolves(dst, r):
                    return f"dangling reference in {path}"
    with open(dst_path, "rb") as f:
        if not f.read(10).startswith(b"MATLAB"):
            return "header is not a MATLAB mat header"
    return None


# ---------------------------------------------------------------------------
# A whole movie
# ---------------------------------------------------------------------------
def movie_mats(movie_dir: str) -> list:
    return sorted(glob.glob(os.path.join(movie_dir, "*_sparse.mat")))


def blank_movie(movie_dir: str, out_dir: str, scan: dict, prescan: dict,
                n_samples: int = PREP_SAMPLES, verify: bool = True) -> dict:
    """Blank every camera of one movie into `out_dir` (same file names) and write its trim.json
    there. `scan` is scan_movie's result for the ORIGINAL mats with the `prescan` parameters.
    Returns the trim record. Raises ValueError when the movie has no build range, or when the
    blanked copy would not prep exactly like the original."""
    rng = build_range(scan["good_start"], scan["good_end"], scan["n_frames"])
    if rng is None:
        raise ValueError("no build range")
    start_ind, end_ind = rng
    os.makedirs(out_dir, exist_ok=True)
    mats = {}
    for src in movie_mats(movie_dir):
        name = os.path.basename(src)
        with h5py.File(src, "r") as f:
            n = len(f[FRAMES][0])
        keep = frames_to_keep(n, start_ind, end_ind, n_samples)
        dst = os.path.join(out_dir, name)
        counts = blank_mat(src, dst, keep)
        if verify:
            problem = check_blanked(src, dst, keep)
            if problem:
                raise ValueError(f"{name}: {problem}")
        mats[name] = dict(counts, kept=merge_ranges(keep),
                          bytes_original=os.path.getsize(src), bytes=os.path.getsize(dst))
    if verify:
        again = scan_movie(out_dir, prescan["pixel_threshold"], prescan["blob_ratio"],
                           prescan["blob_distance"], prescan["min_edge_margin"],
                           prescan["min_cams_in_frame"])
        if "error" in again:
            raise ValueError(f"the blanked movie cannot be scanned: {again['error']}")
        if (again["good_start"], again["good_end"]) != (scan["good_start"], scan["good_end"]):
            raise ValueError(f"the blanked movie scans to frames "
                             f"[{again['good_start']}, {again['good_end']}) instead of "
                             f"[{scan['good_start']}, {scan['good_end']})")
    record = {
        "schema": SCHEMA,
        "made_by": "code/sparse_trim.py",
        "made_at": dt.datetime.now().isoformat(timespec="seconds"),
        "prescan": dict(prescan),
        "n_frames": scan["n_frames"],
        "good_start": scan["good_start"],
        "good_end": scan["good_end"],
        "start_ind": start_ind,
        "end_ind": end_ind,
        "read_range": list(build_read_range(start_ind, end_ind)),
        "mirror_check_samples": n_samples,
        "mats": mats,
    }
    with open(os.path.join(out_dir, TRIM_FILE), "w") as f:
        json.dump(record, f, indent=1)
    return record


# ---------------------------------------------------------------------------
# Guards, for prep on the cluster
# ---------------------------------------------------------------------------
def load_trim(movie_dir: str) -> "dict | None":
    path = os.path.join(movie_dir, TRIM_FILE)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def trim_problem(movie_dir: str, start_ind: int, end_ind: int) -> "str | None":
    """Why prep must not build this movie over [start_ind, end_ind], or None. A movie without
    trim.json was never blanked and is always fine."""
    trim = load_trim(movie_dir)
    if trim is None:
        return None
    if (trim.get("start_ind"), trim.get("end_ind")) != (start_ind, end_ind):
        return (f"the PC that blanked it worked out frames {trim.get('start_ind')}-"
                f"{trim.get('end_ind')}, prep here frames {start_ind}-{end_ind}")
    first, last = build_read_range(start_ind, end_ind)
    for path in movie_mats(movie_dir):
        entry = trim.get("mats", {}).get(os.path.basename(path))
        if entry is None:
            return f"{os.path.basename(path)} is not in {TRIM_FILE}"
        wanted = range(max(first, 0), min(last, entry["n_frames"] - 1) + 1)
        if any(not in_ranges(i, entry["kept"]) for i in wanted):
            return f"{os.path.basename(path)} lacks frames the build reads"
    return None


def sample_problem(movie_dir: str, n_samples: int = PREP_SAMPLES) -> "str | None":
    """Why the mirror check would read blanked frames of this movie, or None."""
    trim = load_trim(movie_dir)
    if trim is None:
        return None
    for path in movie_mats(movie_dir):
        entry = trim.get("mats", {}).get(os.path.basename(path))
        if entry is None:
            return f"{os.path.basename(path)} is not in {TRIM_FILE}"
        missing = [i for i in sample_frame_indices(entry["n_frames"], n_samples)
                   if not in_ranges(i, entry["kept"])]
        if missing:
            return (f"{os.path.basename(path)}: {len(missing)} of the frames the mirror check "
                    f"reads were blanked")
    return None


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    movie_dir, out_dir = sys.argv[1:]
    p = PRESCAN_DEFAULTS
    scan = scan_movie(movie_dir, p["pixel_threshold"], p["blob_ratio"], p["blob_distance"],
                      p["min_edge_margin"], p["min_cams_in_frame"])
    if "error" in scan:
        sys.exit(scan["error"])
    if scan["good_end"] - scan["good_start"] < p["min_intersection"]:
        sys.exit(f"prescan BAD: longest run {scan['good_end'] - scan['good_start']} frames")
    record = blank_movie(movie_dir, out_dir, scan, p)
    for name, m in record["mats"].items():
        print(f"{name}: kept {m['kept_frames']}/{m['n_frames']} frames, "
              f"{m['bytes_original'] / 1e6:.1f} -> {m['bytes'] / 1e6:.1f} MB")
    print(f"build range {record['start_ind']}-{record['end_ind']}; wrote {out_dir}")


if __name__ == "__main__":
    main()

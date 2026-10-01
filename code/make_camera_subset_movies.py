"""
make_camera_subset_movies.py
============================

Cut camera subsets out of built, verified movies, so that a rig the lab does
not have can be predicted from the one it does -- first of all a 2-camera
(bottom + side) rig, from the 4-camera one.

Every source movie becomes one movie per subset, all over the SAME frames:

    bottom-pairs   the bottom camera with each side camera (3 on the 4-camera
                   rig), plus
    all_cams       every camera: the reference the pairs are compared against,
                   predicted over exactly their frames.

Each subset is written as an ordinary experiment -- a calibration.h5, the
experiment's perturbation.json, and one mov<N>/ per movie holding a dataset h5
-- so the normal predict array runs on it unchanged, exactly as it would on a
real rig of that kind. The tool never writes into the source experiment.

WHY CUTTING EQUALS BUILDING
---------------------------
The MATLAB builder crops every camera around its own blob, and the predictor
prepares every camera on its own (cleaning, time-channel alignment, wing
masks); only the left/right assignment and the triangulation look across
cameras. So a camera's channels in a built 4-camera box are the channels a
build from that camera's mat would give over the same frames.

The subset's calibration.h5 is the kept cameras' rows of the source's, with
the source's lab frame (rotation_matrix) kept -- a 2-camera easyWand would put
lab +z on the bisector of the two cameras, and angles would stop being
comparable with the 4-camera result -- and bottom_camera re-indexed into the
subset.

THE WINDOW
----------
The longest run of built frames in which EVERY camera sees a single, whole fly
(the prescan's masks, read from prescan_cam_validity.npz, or re-scanned from
the mats for movies built before it). All subsets share it, so the only
difference between them is the cameras; a pair cut from a frame where one of
its cameras sees a cut fly would not be a movie a 2-camera rig could produce.
Runs shorter than --min-frames (prep's own floor, 500) are skipped.

WHAT IS WRITTEN
---------------
    <out>/<experiment>/<subset>/calibration.h5
    <out>/<experiment>/<subset>/perturbation.json       (when the source has one)
    <out>/<experiment>/<subset>/mov<N>/
        mov_<N>_<start>_<end>_ds_<tc>tc_<tj>tj.h5      window's raw frames in the
                                                       name, so frame 0 is still
                                                       the camera trigger
        *_sparse.mat                                   symlinks to the kept
                                                       cameras' mats (trigger info,
                                                       camera order)
        prescan_cam_validity.npz
        derived_from.json                              where every frame came from
    manifests/sim_<experiment>_<subset>.txt            this run's movies

<experiment> is the source's path below inference_datasets/ (shalev/21to40),
and it is joined with "_" in manifest and run names, because experiment
basenames repeat (21to40 is both Shalev and amitai_dark). Prep refuses any dir
holding derived_from.json: its mats are the source's, and a flip there would
rewrite them.

A movie already cut from the same source file over the same window is left as
it is; anything else is rewritten, and its saved_box_dir cache removed.

USAGE
-----
    # the movies of a verify-passed manifest (never a glob: build writes h5s
    # for movies that then fail verify):
    .env/bin/python code/make_camera_subset_movies.py manifests/<good movies>.txt

    # a pilot, and submit one predict array per subset:
    PREDICT_SBATCH_ARGS="-p catfish --gres=gpu:l4:1 --mem=96g --cpus-per-task=12" \\
    .env/bin/python code/make_camera_subset_movies.py <manifest> --movies mov22,mov24 \\
        --submit --predict-config predict_configurations/config_candidates.json

    # what would happen, writing nothing:
    .env/bin/python code/make_camera_subset_movies.py <manifest> --dry-run
"""

import argparse
import datetime
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from process_experiment import (CAM_VALIDITY_SIDECAR, DEFAULT_MIN_INTERSECTION,
                                DERIVED_FROM_FILE, check_build_complete,
                                find_movie_h5, parse_h5_range, parse_movie_num)
from scan_sparse_movies import (DEFAULT_MIN_EDGE_MARGIN, _longest_run,
                                scan_movie)
from find_mirror_cam import _cam_name as camera_name
from utils import (PredictConfig, declare_bottom_camera, load_bottom_camera,
                   resolve_perturbation_path)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(REPO_ROOT, "inference_datasets", "simulated")
REFERENCE_SUBSET = "all_cams"
SUBSET_PRESETS = ("bottom-pairs",)
TIME_CHANNELS = 3                   # box channels per camera
FRAMES_PER_READ = 100               # box frames read (and written) at a time
PER_CAMERA_CALIBRATION = ("K_matrices", "camera_centers", "camera_matrices",
                          "inv_camera_matrices", "rotation_matrices",
                          "translations")
SHARED_CALIBRATION = ("rotation_matrix",)
# The prescan settings prep uses by default, for movies without a sidecar.
PRESCAN_DEFAULTS = {"pixel_threshold": 50, "blob_ratio": 0.30,
                    "blob_distance": 100.0,
                    "min_edge_margin": DEFAULT_MIN_EDGE_MARGIN}


class Refused(Exception):
    """A source movie this tool will not cut; the message says why."""


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------
def experiment_path(experiment_dir):
    """The experiment's path below inference_datasets/, or its basename."""
    parts = os.path.abspath(experiment_dir).split(os.sep)
    if "inference_datasets" in parts:
        i = len(parts) - 1 - parts[::-1].index("inference_datasets")
        return os.sep.join(parts[i + 1:])
    return parts[-1]


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def read_source(movie_dir):
    """Everything about one source movie that the cut needs. Raises Refused."""
    movie_dir = os.path.abspath(movie_dir)
    if os.path.isfile(os.path.join(movie_dir, DERIVED_FROM_FILE)):
        raise Refused("it was itself cut out of another movie")
    movie_num = parse_movie_num(movie_dir)
    h5 = find_movie_h5(movie_dir)
    if movie_num is None or h5 is None:
        raise Refused("no built dataset h5 in a mov<N> dir")
    rng = parse_h5_range(h5)
    tags = re.search(r"_ds_(\d+)tc_(\d+)tj\.h5$", os.path.basename(h5))
    if rng is None or tags is None:
        raise Refused(f"cannot read the frame range from {os.path.basename(h5)}")
    with h5py.File(h5, "r") as f:
        partial = check_build_complete(f)
        if partial is not None:
            raise Refused(f"its build is incomplete ({partial})")
        cropzone = f["cropzone"][:]
        n_box, n_cams = cropzone.shape[:2]
        if f["box"].shape[1] != n_cams * TIME_CHANNELS:
            raise Refused(f"box has {f['box'].shape[1]} channels for {n_cams} "
                          f"cameras")
    mats = sorted(glob.glob(os.path.join(movie_dir, "*_sparse.mat")))
    if len(mats) != n_cams:
        raise Refused(f"{len(mats)} *_sparse.mat for a {n_cams}-camera box")
    try:
        calibration = PredictConfig.resolve_calibration_path(h5)
        bottom = load_bottom_camera(calibration)
    except SystemExit as e:              # no calibration.h5, or a stamp that
        raise Refused(str(e))            # contradicts the camera positions
    if bottom is None:
        raise Refused("its calibration has no bottom camera (e.g. the old "
                      "side-camera rig), and every subset here needs one")
    stat = os.stat(h5)
    return {"movie_dir": movie_dir, "movie_num": movie_num, "h5": h5,
            "h5_size": stat.st_size, "h5_mtime": stat.st_mtime,
            "start_ind": rng[0], "tc": int(tags.group(1)),
            "tj": int(tags.group(2)), "n_box": n_box, "n_cams": n_cams,
            "cropzone": cropzone, "mats": mats,
            "cam_names": [camera_name(m) for m in mats],
            "calibration": calibration, "bottom": bottom,
            "experiment_dir": os.path.dirname(movie_dir),
            "perturbation": resolve_perturbation_path(h5)}


def whole_fly_masks(src):
    """(in_frame, visible, single), each (n_box, n_cams) bool over the BUILT
    frames, plus where they came from and the prescan settings behind them."""
    sidecar = os.path.join(src["movie_dir"], CAM_VALIDITY_SIDECAR)
    shape = (src["n_box"], src["n_cams"])
    if os.path.isfile(sidecar):
        with np.load(sidecar, allow_pickle=False) as z:
            masks = tuple(np.asarray(z[k], dtype=bool)
                          for k in ("in_frame", "visible", "single"))
            params = json.loads(str(z["params"]))
        if all(m.shape == shape for m in masks):
            return masks, CAM_VALIDITY_SIDECAR, params
    # Built before the sidecar existed: scan the mats as prep would have.
    scan = scan_movie(src["movie_dir"], PRESCAN_DEFAULTS["pixel_threshold"],
                      PRESCAN_DEFAULTS["blob_ratio"],
                      PRESCAN_DEFAULTS["blob_distance"],
                      PRESCAN_DEFAULTS["min_edge_margin"], min_cams_in_frame=0)
    if "error" in scan:
        raise Refused(f"prescan failed: {scan['error']}")
    # box frame k is raw 1-based frame start_ind + k (see
    # process_experiment.write_cam_validity_sidecar)
    lo = src["start_ind"] - 1
    masks = tuple(scan[k][lo:lo + src["n_box"]]
                  for k in ("in_frame_mask", "visible_mask", "single_mask"))
    if any(m.shape != shape for m in masks):
        raise Refused(f"the mats cover {masks[0].shape[0]} of the "
                      f"{src['n_box']} built frames")
    return masks, "rescanned from the mats", dict(PRESCAN_DEFAULTS)


def shared_window(src, min_frames):
    """[a, b) box frames where every camera sees a single, whole fly -- the
    longest such run -- with the provenance of the masks. Raises Refused."""
    (in_frame, visible, single), mask_source, params = whole_fly_masks(src)
    cz = src["cropzone"]
    cropped = ~((cz[:, :, 0] == 1) & (cz[:, :, 1] == 1))    # builder's "no blob"
    good = (in_frame & visible & single & cropped).all(axis=1)
    a, b = _longest_run(good)
    if b - a < min_frames:
        raise Refused(f"only {b - a} consecutive frames with every camera "
                      f"seeing the whole fly (need {min_frames})")
    return (a, b), {"in_frame": in_frame[a:b], "visible": visible[a:b],
                    "single": single[a:b]}, mask_source, params


def camera_subsets(src, preset):
    """[(subset name, camera indices in ascending order)], the reference
    last. Ascending is the order a real rig's sorted mats would give."""
    names, bottom = src["cam_names"], src["bottom"]
    if preset != "bottom-pairs":
        raise ValueError(preset)
    subsets = [tuple(sorted((bottom, side)))
               for side in range(src["n_cams"]) if side != bottom]
    out = [("_".join(names[c] for c in cams), cams) for cams in subsets]
    return out + [(REFERENCE_SUBSET, tuple(range(src["n_cams"])))]


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def write_calibration(src, cams, dest):
    """The kept cameras' calibration, the source's lab frame, and the bottom
    camera re-indexed into the subset. MATLAB wrote every per-camera array
    with the camera LAST as h5py reads it; anything unrecognised is refused
    rather than guessed at."""
    tmp = dest + ".tmp"
    with h5py.File(src["calibration"], "r") as f, h5py.File(tmp, "w") as g:
        for key in f:
            d = f[key]
            if key in PER_CAMERA_CALIBRATION:
                if d.shape[-1] != src["n_cams"]:
                    raise Refused(f"{key} {d.shape}: expected the camera last")
                g.create_dataset(key, data=d[()][..., list(cams)])
            elif key in SHARED_CALIBRATION:
                g.create_dataset(key, data=d[()])
            elif key != "bottom_camera":
                raise Refused(f"unknown calibration dataset {key!r}")
    os.replace(tmp, dest)
    declare_bottom_camera(dest, str(cams.index(src["bottom"])))


def movie_record(src, name, cams, window, mask_source, params, reference_dir):
    a, b = window
    start = src["start_ind"] + a
    return {
        "tool": "code/make_camera_subset_movies.py",
        "git_commit": git_commit(),
        "made_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": {"experiment": experiment_path(src["experiment_dir"]),
                   "movie_dir": src["movie_dir"], "box_h5": src["h5"],
                   "box_h5_size": src["h5_size"],
                   "box_h5_mtime": src["h5_mtime"],
                   "calibration": src["calibration"],
                   "n_cams": src["n_cams"], "n_box_frames": src["n_box"]},
        "subset": {"name": name, "cameras": list(cams),
                   "camera_names": [src["cam_names"][c] for c in cams],
                   "mats": [os.path.basename(src["mats"][c]) for c in cams],
                   "bottom_camera_in_source": src["bottom"],
                   "bottom_camera_in_subset": cams.index(src["bottom"]),
                   "is_reference": name == REFERENCE_SUBSET,
                   "reference_movie_dir": reference_dir},
        "window": {"rule": "every source camera sees a single, whole fly",
                   "masks_from": mask_source, "prescan_params": params,
                   "source_box_frames": [a, b],
                   "raw_frames_1based": [start, start + (b - a) - 1],
                   "n_frames": b - a},
    }


def is_current(movie_out, record):
    """Already cut from this same source file over this same window?"""
    path = os.path.join(movie_out, DERIVED_FROM_FILE)
    if not (os.path.isfile(path) and find_movie_h5(movie_out)):
        return False
    with open(path) as f:
        old = json.load(f)
    if "state" in old:                   # a cut that never finished
        return False
    keys = ("box_h5", "box_h5_size", "box_h5_mtime")
    return (all(old["source"].get(k) == record["source"][k] for k in keys)
            and old["subset"]["cameras"] == record["subset"]["cameras"]
            and old["window"]["source_box_frames"]
            == record["window"]["source_box_frames"])


def clear_movie_dir(movie_out):
    """Remove what a previous cut left: its dataset h5 (find_movie_h5 must
    find exactly the new one) and the predictor's box cache, which would
    otherwise be loaded in place of the new frames."""
    for old in glob.glob(os.path.join(movie_out, "mov_*_ds_*tc_*tj.h5")):
        os.remove(old)
    shutil.rmtree(os.path.join(movie_out, "saved_box_dir"), ignore_errors=True)
    shutil.rmtree(os.path.join(movie_out, ".build_tmp"), ignore_errors=True)


def write_movies(src, jobs, window):
    """Write every job's dataset h5 from ONE pass over the source box.
    jobs: [(movie_out, cams, h5_name)]. Each file is staged in the movie's
    .build_tmp/ and moved into place only when complete, like a build."""
    a, b = window
    n = b - a
    opened = []
    try:
        for movie_out, cams, h5_name in jobs:
            stage = os.path.join(movie_out, ".build_tmp")
            os.makedirs(stage, exist_ok=True)
            g = h5py.File(os.path.join(stage, h5_name), "w")
            channels = [c * TIME_CHANNELS + t for c in cams
                        for t in range(TIME_CHANNELS)]
            nc = len(cams)
            g.create_dataset("box", shape=(n, len(channels), 192, 192),
                             dtype="float32", chunks=(1, len(channels), 192, 192),
                             maxshape=(None, len(channels), 192, 192),
                             compression="gzip", compression_opts=1)
            g.create_dataset("cropzone",
                             data=src["cropzone"][a:b][:, list(cams)],
                             chunks=(1, nc, 2), maxshape=(None, nc, 2),
                             compression="gzip", compression_opts=1)
            # What a fresh build of these frames writes: its loop index,
            # 1 + time_jump onward, the same for every camera.
            frame_inds = np.arange(1 + src["tj"], n + 1 + src["tj"],
                                   dtype=np.uint16)
            g.create_dataset("frameInds",
                             data=np.repeat(frame_inds[:, None, None], nc, axis=1),
                             chunks=(1, nc, 1), maxshape=(None, nc, 1),
                             compression="gzip", compression_opts=1)
            start = src["start_ind"] + a
            g.create_dataset("best_frames_mov_idx", data=np.stack([
                np.full(n, float(src["movie_num"])),
                np.arange(start, start + n, dtype=float)]))
            opened.append((g, channels))
        with h5py.File(src["h5"], "r") as f:
            box = f["box"]
            for lo in range(a, b, FRAMES_PER_READ):
                hi = min(lo + FRAMES_PER_READ, b)
                block = box[lo:hi]
                for g, channels in opened:
                    g["box"][lo - a:hi - a] = block[:, channels]
    finally:
        for g, _ in opened:
            g.close()
    for movie_out, _, h5_name in jobs:
        os.replace(os.path.join(movie_out, ".build_tmp", h5_name),
                   os.path.join(movie_out, h5_name))
        shutil.rmtree(os.path.join(movie_out, ".build_tmp"), ignore_errors=True)


def link_mats(src, cams, movie_out):
    for c in cams:
        target = os.path.realpath(src["mats"][c])
        link = os.path.join(movie_out, os.path.basename(src["mats"][c]))
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(target, link)


def write_validity(masks, cams, movie_out, record, params):
    meta = dict(params)
    meta.update({"start_ind": record["window"]["raw_frames_1based"][0],
                 "n_box_frames": record["window"]["n_frames"],
                 "min_cams_in_frame": len(cams), "n_cams": len(cams),
                 "cut_from": record["source"]["box_h5"]})
    np.savez_compressed(
        os.path.join(movie_out, CAM_VALIDITY_SIDECAR),
        mat_names=np.array(record["subset"]["mats"]), params=json.dumps(meta),
        **{k: v[:, list(cams)] for k, v in masks.items()})


def cut_movie(src, preset, out_root, min_frames, dry_run):
    """Cut one source movie into its subsets. Returns [(subset, movie_out,
    state)] with state 'written' / 'up to date' / 'would write'."""
    window, masks, mask_source, params = shared_window(src, min_frames)
    subsets = camera_subsets(src, preset)
    exp_out = os.path.join(out_root, experiment_path(src["experiment_dir"]))
    movie_name = os.path.basename(src["movie_dir"])
    reference_dir = os.path.join(exp_out, REFERENCE_SUBSET, movie_name)
    a, b = window
    h5_name = (f"mov_{src['movie_num']}_{src['start_ind'] + a}_"
               f"{src['start_ind'] + b - 1}_ds_{src['tc']}tc_{src['tj']}tj.h5")
    results, jobs, records = [], [], {}
    for name, cams in subsets:
        movie_out = os.path.join(exp_out, name, movie_name)
        record = movie_record(src, name, cams, window, mask_source, params,
                              reference_dir)
        if is_current(movie_out, record):
            results.append((name, movie_out, "up to date"))
            continue
        results.append((name, movie_out, "would write" if dry_run else "written"))
        jobs.append((movie_out, cams, h5_name))
        records[movie_out] = (name, cams, record)
    if dry_run or not jobs:
        return window, mask_source, results
    for movie_out, cams, _ in jobs:
        os.makedirs(movie_out, exist_ok=True)
        clear_movie_dir(movie_out)
        # Invalidate first: a cut that dies partway must not look current.
        marker = os.path.join(movie_out, DERIVED_FROM_FILE)
        if os.path.exists(marker):
            os.remove(marker)
    for movie_out, cams, _ in jobs:
        # derived_from.json is only the "current?" marker once the h5 is in
        # place; the refusal guard in prep needs a marker from the start.
        with open(os.path.join(movie_out, DERIVED_FROM_FILE), "w") as f:
            json.dump({"state": "being cut", **records[movie_out][2]}, f,
                      indent=2)
    write_movies(src, jobs, window)
    for movie_out, cams, _ in jobs:
        name, _, record = records[movie_out]
        subset_dir = os.path.dirname(movie_out)
        write_calibration(src, cams, os.path.join(subset_dir, "calibration.h5"))
        pert = src["perturbation"]
        if pert:
            level = (movie_out if os.path.dirname(pert) == src["movie_dir"]
                     else subset_dir)
            shutil.copy2(pert, os.path.join(level, os.path.basename(pert)))
        link_mats(src, cams, movie_out)
        write_validity(masks, cams, movie_out, record, params)
        with open(os.path.join(movie_out, DERIVED_FROM_FILE), "w") as f:
            json.dump(record, f, indent=2)
    return window, mask_source, results


# ---------------------------------------------------------------------------
# Manifests and prediction
# ---------------------------------------------------------------------------
def run_name(experiment, subset):
    """The predict run (and manifest) name of one subset of one experiment:
    predictions land in predict_output/<run name>/."""
    return "sim_" + experiment.replace(os.sep, "_") + "_" + subset


def write_manifests(made, dry_run):
    """manifests/sim_<experiment>_<subset>.txt per (experiment, subset), this
    run's movies in order. Returns [(run name, manifest path, n movies)]."""
    groups = {}
    for exp, subset, movie_out in made:
        groups.setdefault((exp, subset), []).append(movie_out)
    out = []
    for (exp, subset), dirs in sorted(groups.items()):
        run = run_name(exp, subset)
        path = os.path.join(REPO_ROOT, "manifests", run + ".txt")
        if not dry_run:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.writelines(d + "\n" for d in dirs)
        out.append((run, path, len(dirs)))
    return out


def submit(manifests, predict_config, concurrency):
    """One predict array per subset, named after it, as pipeline.sh submits
    its own: PREDICT_SBATCH_ARGS is word-split onto the command line."""
    extra = shlex.split(os.environ.get("PREDICT_SBATCH_ARGS", ""))
    for run, path, n in manifests:
        cmd = (["sbatch", "-J", run, f"--array=0-{n - 1}%{concurrency}"]
               + extra + ["sbatch_files/predict_array.sh",
                          os.path.relpath(path, REPO_ROOT), predict_config])
        print("  " + " ".join(shlex.quote(c) for c in cmd))
        r = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        print("    " + (r.stdout.strip() or r.stderr.strip()))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def read_manifest(path, only):
    with open(path) as f:
        dirs = [line.strip().rstrip(os.sep) for line in f if line.strip()]
    if only:
        wanted = {m.strip() for m in only.split(",") if m.strip()}
        dirs = [d for d in dirs if os.path.basename(d) in wanted]
        missing = wanted - {os.path.basename(d) for d in dirs}
        if missing:
            sys.exit(f"not in {path}: {', '.join(sorted(missing))}")
    return [d if os.path.isabs(d) else os.path.join(REPO_ROOT, d) for d in dirs]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest",
                    help="verify-passed movie dirs, one per line (a good_movies "
                         "manifest, or one made from process_report PASS lines)")
    ap.add_argument("--subsets", default="bottom-pairs", choices=SUBSET_PRESETS,
                    help="which camera subsets to cut, besides the all_cams "
                         "reference (default: bottom-pairs)")
    ap.add_argument("--movies", default="",
                    help="only these movie dirs of the manifest, e.g. mov22,mov24")
    ap.add_argument("--min-frames", type=int, default=DEFAULT_MIN_INTERSECTION,
                    help="shortest shared window worth predicting (default: "
                         f"{DEFAULT_MIN_INTERSECTION}, prep's own floor)")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="root the subset experiments go under (default: "
                         "inference_datasets/simulated)")
    ap.add_argument("--submit", action="store_true",
                    help="submit one predict array per subset afterwards")
    ap.add_argument("--predict-config",
                    help="predict config for --submit; which model set the "
                         "subsets are predicted with is a choice, so there is "
                         "no default")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="array tasks running at once, per subset (default 8)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report windows and what would be written; write nothing")
    args = ap.parse_args()
    if args.submit and not args.predict_config:
        ap.error("--submit needs --predict-config")
    if args.submit and args.dry_run:
        ap.error("--submit and --dry-run exclude each other")

    made, refused = [], []
    for movie_dir in read_manifest(args.manifest, args.movies):
        label = os.path.relpath(movie_dir, REPO_ROOT)
        try:
            src = read_source(movie_dir)
            window, mask_source, results = cut_movie(
                src, args.subsets, os.path.abspath(args.out), args.min_frames,
                args.dry_run)
        except Refused as e:
            print(f"  {label}: SKIPPED -- {e}")
            refused.append(label)
            continue
        a, b = window
        print(f"  {label}: box frames [{a}, {b}) of {src['n_box']} "
              f"({b - a} frames, raw {src['start_ind'] + a}.."
              f"{src['start_ind'] + b - 1}; masks {mask_source}); bottom "
              f"camera {src['cam_names'][src['bottom']]}")
        exp = experiment_path(src["experiment_dir"])
        for subset, movie_out, state in results:
            print(f"      {subset:<12} {state:<12} "
                  f"{os.path.relpath(movie_out, REPO_ROOT)}")
            made.append((exp, subset, movie_out))

    n_sources = len({(exp, os.path.basename(out)) for exp, _, out in made})
    print(f"\n{len(made)} subset movie(s) from {n_sources} source movie(s); "
          f"{len(refused)} skipped.")
    if not made:
        sys.exit(1)
    manifests = write_manifests(made, args.dry_run)
    for run, path, n in manifests:
        print(f"  {'would write' if args.dry_run else 'manifest'}: "
              f"{os.path.relpath(path, REPO_ROOT)}  ({n} movies)  run {run}")
    if args.submit:
        print("\nsubmitting:")
        submit(manifests, args.predict_config, args.concurrency)


if __name__ == "__main__":
    main()

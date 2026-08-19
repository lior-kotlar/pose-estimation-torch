"""
import_old_experiment.py
========================

Import an experiment that was already processed by an OLDER version of this
pipeline into the standard `inference_datasets/` layout, so the current
pipeline can be run over it unchanged.

The per-experiment facts (which archive, which easyWand won, what is
superseded) live in a JSON spec under `import_specs/`, not in this file --
`inference_datasets/` is gitignored, so a spec is the only durable, committed
record of why a given experiment was imported the way it was.

    .env/bin/python code/import_old_experiment.py import_specs/<name>.json

WHAT IT PRODUCES
----------------
    <dest>/
      easywand/
        <install_as>                       the calibration source for this experiment
        _superseded/                       the easyWand variants that were rejected
      old_predictions/
        mov<N>/points_*.npy                the old pipeline's 3D output, kept for
                                           old-vs-new comparison
        manifest.json                      per-movie old build range + provenance
      <batch>/                             e.g. 1to20, 21to40
        calibration.h5                     copy of the archive's verified calibration
        mov<N>/
          mov<N>_cam<mirror>_sparse.mat    REAL file, converted v7 -> v7.3
          mov<N>_cam*_sparse.mat           HARD LINKS to the archive
          mov<N>_*.mp4                     HARD LINK (source renders only, if any)
          README_mov<N>.txt                copy
      README.md                            the gotchas, rendered from the spec

WHY HARD LINKS
--------------
The non-mirror mats and the mp4s are bit-identical to the archive copies and
are never written to by the pipeline, so a second directory entry costs no
space. Hard links (not symlinks) because the archive is usually pruned
afterwards: with a hard link both names are equally real and deleting one does
not break the other, whereas a symlink farm would be left dangling.

WHY THE MIRROR CAM IS A REAL FILE
---------------------------------
In these old archives the mirror cam is typically ALREADY flipped
(`metaData.isFlipped == 1`) by a flip script whose bare `save` downgraded the
file to the v7 MAT container. h5py cannot open v7, which breaks three things:
the prescan (`scan_sparse_movies.py`), the MATLAB builder's `matfile` partial
loading, and `utils.get_trigger_frame_info` (which reads the FIRST sparse mat
alphabetically). So that cam is converted to v7.3 by
`matlab/resave_sparse_v73.m` into a NEW file here, leaving the archive
original untouched. That script refuses any mat whose `isFlipped` is not set,
so an unflipped archive fails loudly instead of importing silently.

    ==> Everything downstream must then run with --skip-flip. <==

USAGE
-----
    .env/bin/python code/import_old_experiment.py <spec.json> --dry-run
    .env/bin/python code/import_old_experiment.py <spec.json>
    .env/bin/python code/import_old_experiment.py <spec.json> --skip-matlab

The MATLAB conversion runs last, once, over every mirror-cam mat together
(MATLAB startup is ~90 s, so batching matters). Re-running is safe: existing
outputs are left alone unless --force, which makes it the natural way to fold
a later delivery of the same experiment into the same tree.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESAVE_SCRIPT_DIR = os.path.join(REPO_ROOT, "matlab")
MATLAB_BIN = os.environ.get("MATLAB_BIN", "matlab")

OLD_PRED_NPY_GLOBS = ("points_3D_*.npy", "points_ensemble_*.npy")


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------
def load_spec(path):
    """Read the experiment spec and resolve its paths against the repo root."""
    with open(path) as f:
        spec = json.load(f)
    for key in ("source_archive", "dest"):
        if key not in spec:
            sys.exit(f"spec is missing required key '{key}': {path}")
        if not os.path.isabs(spec[key]):
            spec[key] = os.path.join(REPO_ROOT, spec[key])
    spec.setdefault("label", os.path.basename(spec["dest"]))
    spec.setdefault("batch_size", 20)
    spec.setdefault("mirror_cam", "cam1")
    spec.setdefault("source_calibration_h5", "calibration.h5")
    ew = spec.setdefault("easywand", {})
    ew.setdefault("dir", "calibration")
    ew.setdefault("superseded_globs", [])
    ew.setdefault("superseded_dirs", [])
    if ew.get("choose"):
        ew.setdefault("install_as", ew["choose"])
    return spec


def _lines(spec, key):
    """A spec note field, given as a list of lines or a single string."""
    v = spec.get(key) or []
    return "\n".join(v) if isinstance(v, list) else str(v)


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------
def movie_num(d):
    m = re.match(r"^mov(\d+)$", os.path.basename(d), re.IGNORECASE)
    return int(m.group(1)) if m else None


def discover_movies(src):
    out = []
    for name in sorted(os.listdir(src)):
        p = os.path.join(src, name)
        if not os.path.isdir(p):
            continue
        n = movie_num(p)
        if n is None:
            continue
        mats = sorted(glob.glob(os.path.join(p, "*_sparse.mat")))
        if len(mats) != 4:
            print(f"  WARNING: {name} has {len(mats)} *_sparse.mat (expected 4); skipping")
            continue
        out.append((n, p))
    out.sort()
    return out


def batch_name(nums):
    return f"{nums[0]}to{nums[-1]}"


def link_or_copy(src, dst, dry, force, mode="link"):
    """Hard-link (default) or copy src -> dst. Returns an action string."""
    if os.path.exists(dst):
        if not force:
            return "exists"
        if not dry:
            os.remove(dst)
    if dry:
        return f"would {mode}"
    if mode == "link":
        try:
            os.link(src, dst)
            return "linked"
        except OSError as e:
            # Cross-device or a filesystem without hard links: fall back rather
            # than abort, but say so loudly -- it costs real disk.
            print(f"    hard link failed ({e}); copying instead")
            shutil.copy2(src, dst)
            return "copied (link failed)"
    shutil.copy2(src, dst)
    return "copied"


def old_build_range(movie_dir):
    """(start_ind, end_ind, h5_name) from the old build h5 name, or Nones.

    The old h5 is `movie_<N>_<start>_<end>_ds_3tc_7tj.h5` -- note `movie_`,
    where today's builder writes `mov_`. Those indices are 1-based into the
    raw mat, the same convention the new build uses, so they are what aligns
    the old and new point arrays for comparison.
    """
    for p in sorted(glob.glob(os.path.join(movie_dir, "movie_*_ds_*tc_*tj.h5"))):
        m = re.match(r"^movie_\d+_(\d+)_(\d+)_ds_", os.path.basename(p))
        if m:
            return int(m.group(1)), int(m.group(2)), os.path.basename(p)
    return None, None, None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="path to the experiment spec JSON (import_specs/*.json)")
    ap.add_argument("--src", default=None, help="override the spec's source_archive")
    ap.add_argument("--dest", default=None, help="override the spec's dest")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override the spec's batch_size")
    ap.add_argument("--copy-mats", action="store_true",
                    help="copy the non-mirror cams instead of hard-linking")
    ap.add_argument("--skip-matlab", action="store_true",
                    help="stage everything but don't run the mirror-cam conversion")
    ap.add_argument("--force", action="store_true",
                    help="overwrite files that already exist in dest")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.spec):
        sys.exit(f"spec not found: {args.spec}")
    spec = load_spec(args.spec)
    src = args.src or spec["source_archive"]
    dest = args.dest or spec["dest"]
    batch_size = args.batch_size or spec["batch_size"]
    mirror = spec["mirror_cam"]
    dry = args.dry_run

    if not os.path.isdir(src):
        sys.exit(f"source not found: {src}")

    movies = discover_movies(src)
    if not movies:
        sys.exit(f"no mov<N>/ dirs with 4 sparse mats under {src}")
    nums = [n for n, _ in movies]
    print(f"spec:   {args.spec}  ({spec['label']})")
    print(f"source: {src}")
    print(f"dest:   {dest}")
    print(f"movies: {len(movies)}  (mov{nums[0]}..mov{nums[-1]})")
    print(f"mode:   {'DRY RUN' if dry else 'APPLY'}, "
          f"non-mirror cams {'copied' if args.copy_mats else 'hard-linked'}, "
          f"mirror cam '{mirror}' converted to v7.3, batch size {batch_size}\n")

    batches = [movies[i:i + batch_size] for i in range(0, len(movies), batch_size)]

    def mkdir(p):
        if not dry:
            os.makedirs(p, exist_ok=True)

    mkdir(dest)
    convert_pairs = []
    manifest = {}
    stats = {}

    # ---------------- movie batches ----------------
    for b in batches:
        bname = batch_name([n for n, _ in b])
        bdir = os.path.join(dest, bname)
        print(f"=== batch {bname}  ({len(b)} movies) -> {bdir}")
        mkdir(bdir)

        calib_src = os.path.join(src, spec["source_calibration_h5"])
        if os.path.isfile(calib_src):
            act = link_or_copy(calib_src, os.path.join(bdir, "calibration.h5"),
                               dry, args.force, mode="copy")
            print(f"  calibration.h5: {act}")
        else:
            print(f"  WARNING: {calib_src} not found")

        for n, mdir in b:
            mname = f"mov{n}"
            tgt = os.path.join(bdir, mname)
            mkdir(tgt)
            actions = []
            for mat in sorted(glob.glob(os.path.join(mdir, "*_sparse.mat"))):
                base = os.path.basename(mat)
                if f"_{mirror}_" in base:
                    continue        # the mirror cam is converted, not linked
                a = link_or_copy(mat, os.path.join(tgt, base), dry, args.force,
                                 mode="copy" if args.copy_mats else "link")
                actions.append(f"{base.split('_')[-2]}:{a}")
                stats[a] = stats.get(a, 0) + 1
            # Anchor the mp4 glob on the movie name. Some archives also hold a
            # render from the old pipeline under a generic name (e.g.
            # `analisys_mp4.mp4`); that is superseded output, not source
            # footage, so it stays in the archive. Some movies have no mp4 at
            # all -- fine, nothing in the pipeline reads it.
            for mp4 in sorted(glob.glob(os.path.join(mdir, f"{mname}_*.mp4"))):
                a = link_or_copy(mp4, os.path.join(tgt, os.path.basename(mp4)),
                                 dry, args.force,
                                 mode="copy" if args.copy_mats else "link")
                actions.append(f"mp4:{a}")
                stats[a] = stats.get(a, 0) + 1
            rd = os.path.join(mdir, f"README_{mname}.txt")
            if os.path.isfile(rd):
                link_or_copy(rd, os.path.join(tgt, os.path.basename(rd)), dry,
                             args.force, mode="copy")
            # mirror cam: queued for the MATLAB v7 -> v7.3 pass
            c_src = sorted(glob.glob(os.path.join(mdir, f"*_{mirror}_sparse.mat")))
            if not c_src:
                actions.append(f"{mirror}:MISSING")
            else:
                c_dst = os.path.join(tgt, os.path.basename(c_src[0]))
                if os.path.exists(c_dst) and not args.force:
                    actions.append(f"{mirror}:exists")
                else:
                    convert_pairs.append((c_src[0], c_dst))
                    actions.append(f"{mirror}:queued-for-v7.3")
            print(f"  {mname:>6}: " + "  ".join(actions))

            s_i, e_i, h5n = old_build_range(mdir)
            manifest[mname] = {
                "batch": bname,
                "old_build_h5": h5n,
                "old_start_ind": s_i,
                "old_end_ind": e_i,
                "old_run_completed": os.path.isfile(os.path.join(mdir, "done.txt")),
                "source_archive_dir": mdir,
                "imported_movie_dir": tgt,
            }

    # ---------------- easyWand ----------------
    ew = spec["easywand"]
    ew_dir = os.path.join(dest, "easywand")
    ew_sup = os.path.join(ew_dir, "_superseded")
    print(f"\n=== easyWand -> {ew_dir}")
    mkdir(ew_dir)
    mkdir(ew_sup)
    cal_src_dir = os.path.join(src, ew["dir"])
    if ew.get("choose"):
        chosen = os.path.join(cal_src_dir, ew["choose"])
        if os.path.isfile(chosen):
            a = link_or_copy(chosen, os.path.join(ew_dir, ew["install_as"]), dry,
                             args.force, mode="copy")
            print(f"  {ew['install_as']}: {a}")
        else:
            print(f"  WARNING: chosen easyWand not found: {chosen}")
    for pat in ew["superseded_globs"]:
        for p in sorted(glob.glob(os.path.join(cal_src_dir, pat))):
            link_or_copy(p, os.path.join(ew_sup, os.path.basename(p)), dry,
                         args.force, mode="copy")
    for dname in ew["superseded_dirs"]:
        s = os.path.join(cal_src_dir, dname)
        d = os.path.join(ew_sup, dname)
        if os.path.isdir(s) and (args.force or not os.path.isdir(d)):
            if dry:
                print(f"  would copy dir {dname}/")
            else:
                if os.path.isdir(d):
                    shutil.rmtree(d)
                shutil.copytree(s, d)
    print(f"  superseded variants -> {ew_sup}")

    # ---------------- old predictions ----------------
    op_dir = os.path.join(dest, "old_predictions")
    print(f"\n=== old predictions -> {op_dir}")
    mkdir(op_dir)
    n_npy = 0
    for mname, meta in manifest.items():
        mdir = meta["source_archive_dir"]
        files = []
        for pat in OLD_PRED_NPY_GLOBS:
            files.extend(sorted(glob.glob(os.path.join(mdir, pat))))
        meta["n_old_npy"] = len(files)
        if not files:
            print(f"  {mname:>6}: no prediction output (nothing to salvage)")
            continue
        t = os.path.join(op_dir, mname)
        mkdir(t)
        for p in files:
            link_or_copy(p, os.path.join(t, os.path.basename(p)), dry,
                         args.force, mode="copy")
        n_npy += len(files)
        print(f"  {mname:>6}: {len(files)} npy  "
              f"(old range {meta['old_start_ind']}-{meta['old_end_ind']})")
    if not dry:
        mpath = os.path.join(op_dir, "manifest.json")
        # Fold into any manifest already there, so a later delivery of the same
        # experiment adds to the record instead of erasing the earlier movies.
        merged = {}
        if os.path.isfile(mpath):
            try:
                with open(mpath) as f:
                    merged = json.load(f).get("movies", {})
            except (json.JSONDecodeError, OSError):
                merged = {}
        merged.update(manifest)
        with open(mpath, "w") as f:
            json.dump({
                "experiment": spec["label"],
                "note": ("3D output from the OLD pipeline, kept for old-vs-new "
                         "comparison. Same joint layout, units and world frame "
                         "as the current pipeline (same easyWand), so points "
                         "are directly comparable in absolute coordinates. "
                         "Align frames via old_start_ind vs the new h5's "
                         "start_ind -- both index the same raw mat."),
                "movies": dict(sorted(merged.items(),
                                      key=lambda kv: int(kv[0][3:]))),
            }, f, indent=2)
    print(f"  total npy salvaged: {n_npy}")

    # ---------------- README ----------------
    readme = os.path.join(dest, "README.md")
    if not dry and (args.force or not os.path.isfile(readme)):
        with open(readme, "w") as f:
            f.write(README_TEMPLATE.format(
                label=spec["label"],
                n=len(movies), lo=nums[0], hi=nums[-1],
                batches=", ".join(batch_name([n for n, _ in b]) for b in batches),
                mirror=mirror,
                easywand=ew.get("install_as", "(none)"),
                src=src,
                spec=os.path.relpath(os.path.abspath(args.spec), REPO_ROOT),
                calibration_notes=_lines(spec, "calibration_notes") or "(none recorded)",
                known_gaps=_lines(spec, "known_gaps") or "(none recorded)"))
        print(f"\nwrote {readme}")

    # ---------------- MATLAB mirror-cam conversion ----------------
    print(f"\n=== {mirror} v7 -> v7.3 conversion: {len(convert_pairs)} file(s)")
    if not convert_pairs:
        print("  nothing to convert")
    elif dry:
        for s, d in convert_pairs[:3]:
            print(f"  would convert {s}\n             -> {d}")
        if len(convert_pairs) > 3:
            print(f"  ... and {len(convert_pairs) - 3} more")
    elif args.skip_matlab:
        print(f"  --skip-matlab: skipped ({mirror} files are MISSING from dest)")
    else:
        tsv = os.path.join(dest, ".mirror_convert_list.tsv")
        with open(tsv, "w") as f:
            for s, d in convert_pairs:
                f.write(f"{s}\t{d}\n")
        cmd = f"addpath('{RESAVE_SCRIPT_DIR}'); resave_sparse_v73('{tsv}')"
        print(f"  running MATLAB over {tsv} (one invocation, ~90 s startup)")
        rc = subprocess.call([MATLAB_BIN, "-batch", cmd])
        if rc != 0:
            sys.exit(f"\nMATLAB conversion FAILED (exit {rc}); "
                     f"dest is incomplete -- fix before running the pipeline")
        os.remove(tsv)
        print("  conversion OK")

    print("\n--- summary ---")
    for k, v in sorted(stats.items()):
        if v:
            print(f"  {k}: {v}")
    print(f"  movies imported: {len(movies)} in {len(batches)} batch(es)")
    if dry:
        print("\n(dry run: nothing was written)")


README_TEMPLATE = """\
# {label}

{n} movies (mov{lo}..mov{hi}), imported from `{src}`
by `code/import_old_experiment.py` using `{spec}`.
Batches: {batches}

## Read this before running the pipeline

**Run with `--skip-flip`.** `{mirror}` is the mirror camera and the source mats are
ALREADY flipped (`metaData.isFlipped == 1`). Flipping again would silently
invert one camera and wreck the triangulation. There is no idempotency check in
the flip step -- this README is the guard.

**`{mirror}` was converted from the v7 MAT container to v7.3** by
`matlab/resave_sparse_v73.m`. The old flip script ended with a bare `save`,
which downgraded the file; h5py cannot open that, which broke the prescan, the
MATLAB builder's `matfile` partial loading, and trigger-frame lookup. The
`{mirror}` files here are real, converted files; the other cams and the mp4s are
**hard links** into the archive.

**Predict under ONE dated run name for the whole experiment**, e.g.
`sbatch -J <experiment>_<date> ...`. The consumer keys on
`os.path.dirname(dir)` in each analysis h5, which is the predict-output run
directory -- so splitting one experiment across several `-J` names makes it
look like several experiments, and an undated name reads as unidentifiable.
Input batching is unrelated and invisible to the consumer; `pipeline.sh` takes
the manifest from the input dir and the run name from `-J`, independently.

**Do not leave old `movie_*_ds_*.h5` build files in a movie dir.**
`predict.py`'s `configure_movie_list` matches any `mov*` + `.h5`, so a stale
build h5 would be queued for prediction alongside the new one. (The prep step
is unaffected: `find_movie_h5` uses the stricter `mov_*_ds_*tc_*tj.h5`.)

## Calibration

`easywand/{easywand}` is the one to use.

{calibration_notes}

`easywand/_superseded/` holds the rejected variants. `calibration.h5` in each
batch dir is a copy of the archive's verified one; the prep step overwrites it
from the easyWand mat, which is fine -- they agree.

## Old predictions

`old_predictions/` holds the old pipeline's 3D output (`points_*.npy`) plus
`manifest.json` with each movie's old build range. To align frames, use
`old_start_ind` against the new h5's `start_ind` -- both are 1-based indices
into the same raw mat.

## Known gaps

{known_gaps}
"""


if __name__ == "__main__":
    main()

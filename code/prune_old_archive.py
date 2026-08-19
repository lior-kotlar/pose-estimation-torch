"""
prune_old_archive.py
====================

Delete the superseded, regenerable output from an old-experiment archive after
it has been imported into the standard layout by `code/import_old_experiment.py`.

Point it at the archive and at that import's `old_predictions/` directory; the
paths come from the same spec used for the import (`import_specs/*.json`).

WHAT GOES
---------
    mov*/saved_box_dir/                   the cropped box,
                                          masks and sizes; a pure intermediate the
                                          predict step rebuilds from the raw mats
    mov*/movie_*_ds_*tc_*tj.h5            the old build h5. Regenerable, AND it
                                          actively breaks the new pipeline:
                                          predict.py's configure_movie_list globs
                                          `mov*` + `.h5`, so a stale build h5 gets
                                          queued for prediction alongside the new one
    mov*/*WINGS_AND_BODY_SAME_MODEL*/     the old per-member ensemble dirs (2D
                                          predictions + per-member 3D), superseded
                                          by the current models
    mov*/*.html                           regenerable plotly viewers

WHAT STAYS
----------
    mov*/*_sparse.mat                     the raw input -- never touched
    mov*/*.mp4                            the raw movie
    mov*/points_*.npy                     the old 3D output (the comparison value)
    mov*/README_*.txt, *.json, *.txt      provenance, tiny

After the prune the archive holds little UNIQUE data: the sparse mats and mp4s
are hard-linked into the import, so those bytes are shared and the archive costs
almost nothing to keep around as a browsable original.

SAFETY
------
Refuses to delete unless every `points_*.npy` in the archive has a byte-identical
counterpart under the import's `old_predictions/`. That is the only irreplaceable
content here, and it is checked by md5 -- not by existence -- before anything is
removed. Run with --dry-run first (it is the default posture: --apply is required
to actually delete).

USAGE
-----
    .env/bin/python code/prune_old_archive.py --spec import_specs/<name>.json
    .env/bin/python code/prune_old_archive.py --spec import_specs/<name>.json --apply

    # or name the two directories explicitly:
    .env/bin/python code/prune_old_archive.py --archive <dir> --salvage <dir>
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DELETE_DIR_GLOBS = ("saved_box_dir", "*WINGS_AND_BODY_SAME_MODEL*")
DELETE_FILE_GLOBS = ("movie_*_ds_*tc_*tj.h5", "*.html")


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f} {u}"
        n /= 1024


def dir_size(p):
    total = 0
    for root, _, files in os.walk(p):
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def verify_salvage(archive, salvage):
    """Every archive points_*.npy must exist under salvage with the same md5.
    Returns (ok, n_checked, problems)."""
    problems = []
    n = 0
    for mdir in sorted(glob.glob(os.path.join(archive, "mov*"))):
        m = os.path.basename(mdir)
        srcs = sorted(glob.glob(os.path.join(mdir, "points_3D_*.npy")) +
                      glob.glob(os.path.join(mdir, "points_ensemble_*.npy")))
        for s in srcs:
            d = os.path.join(salvage, m, os.path.basename(s))
            if not os.path.isfile(d):
                problems.append(f"MISSING in salvage: {m}/{os.path.basename(s)}")
                continue
            if md5(s) != md5(d):
                problems.append(f"MD5 MISMATCH: {m}/{os.path.basename(s)}")
                continue
            n += 1
    return (not problems), n, problems


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=None,
                    help="import spec JSON; --archive/--salvage are read from "
                         "its source_archive and dest (dest/old_predictions)")
    ap.add_argument("--archive", default=None,
                    help="the old-experiment archive to prune")
    ap.add_argument("--salvage", default=None,
                    help="the import's old_predictions/ dir, used to verify "
                         "every npy is safely copied before anything is deleted")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default is a dry run)")
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip the md5 salvage check (NOT recommended)")
    args = ap.parse_args()

    if args.spec:
        with open(args.spec) as f:
            spec = json.load(f)
        def _abs(x):
            return x if os.path.isabs(x) else os.path.join(REPO_ROOT, x)
        args.archive = args.archive or _abs(spec["source_archive"])
        args.salvage = args.salvage or os.path.join(_abs(spec["dest"]),
                                                    "old_predictions")
    if not args.archive or not args.salvage:
        sys.exit("need --spec, or both --archive and --salvage")
    if not os.path.isdir(args.archive):
        sys.exit(f"archive not found: {args.archive}")

    print(f"archive: {args.archive}")
    print(f"salvage: {args.salvage}")
    print(f"mode:    {'APPLY (deleting)' if args.apply else 'DRY RUN'}\n")

    if not args.skip_verify:
        print("=== salvage verification (md5 of every points_*.npy) ===")
        if not os.path.isdir(args.salvage):
            sys.exit(f"salvage dir not found: {args.salvage}\n"
                     f"Run code/import_old_experiment.py first.")
        ok, n, problems = verify_salvage(args.archive, args.salvage)
        print(f"  {n} npy verified identical")
        if not ok:
            print(f"  {len(problems)} PROBLEM(S):")
            for p in problems[:20]:
                print(f"    {p}")
            sys.exit("\nREFUSING TO DELETE -- the salvage is incomplete.")
        print("  OK: every archive npy has a byte-identical salvaged copy\n")

    targets = []          # (path, is_dir, size)
    for mdir in sorted(glob.glob(os.path.join(args.archive, "mov*"))):
        for pat in DELETE_DIR_GLOBS:
            for p in sorted(glob.glob(os.path.join(mdir, pat))):
                if os.path.isdir(p):
                    targets.append((p, True, dir_size(p)))
        for pat in DELETE_FILE_GLOBS:
            for p in sorted(glob.glob(os.path.join(mdir, pat))):
                if os.path.isfile(p):
                    targets.append((p, False, os.path.getsize(p)))

    if not targets:
        print("nothing to delete (already pruned?)")
        return

    by_kind = {}
    for p, is_dir, sz in targets:
        b = os.path.basename(p)
        if b == "saved_box_dir":
            k = "saved_box_dir/"
        elif "WINGS_AND_BODY_SAME_MODEL" in b:
            k = "ensemble member dirs/"
        elif b.endswith(".html"):
            k = "*.html"
        else:
            k = "old build h5"
        e = by_kind.setdefault(k, [0, 0])
        e[0] += 1
        e[1] += sz

    total = sum(sz for _, _, sz in targets)
    print("=== to delete ===")
    for k, (cnt, sz) in sorted(by_kind.items(), key=lambda kv: -kv[1][1]):
        print(f"  {k:<26} {cnt:>5} item(s)  {human(sz):>10}")
    print(f"  {'TOTAL':<26} {len(targets):>5} item(s)  {human(total):>10}")

    if not args.apply:
        print("\n(dry run: nothing deleted; re-run with --apply)")
        return

    print("\n=== deleting ===")
    n_ok = n_err = 0
    freed = 0
    for p, is_dir, sz in targets:
        try:
            if is_dir:
                shutil.rmtree(p)
            else:
                os.remove(p)
            freed += sz
            n_ok += 1
        except OSError as e:
            print(f"  ERROR removing {p}: {e}")
            n_err += 1
    print(f"  removed {n_ok} item(s), freed {human(freed)}, {n_err} error(s)")

    remaining = dir_size(args.archive)
    print(f"\narchive now holds {human(remaining)} "
          f"(most of it hard-linked into the import, so shared not duplicated)")


if __name__ == "__main__":
    main()

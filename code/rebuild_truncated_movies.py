"""Rebuild the 12 movies whose original MATLAB build died partway.

Each is rebuilt with the exact start_ind/end_ind recorded in its experiment's
process_report.txt, through process_experiment.build_one_movie -- so it uses
the same staged-commit path as the pipeline: MATLAB writes into .build_tmp and
the h5 lands only on a zero exit.

Aborts if free space drops below MIN_FREE_GB, since the volume is nearly full.
"""
import os
import shutil
import sys
import time

sys.path.insert(0, "/cs/labs/tsevi/lior.kotlar/pose-estimation-torch/code")
os.chdir("/cs/labs/tsevi/lior.kotlar/pose-estimation-torch")

import h5py
from process_experiment import (build_one_movie, check_build_complete,
                                find_movie_h5, verify_one_movie,
                                find_calibration_h5)
from pipeline_timing import record as record_timing

MIN_FREE_GB = 6

TARGETS = [
    ("roni_dark/2023_08_06_40ms/1to30",     14,   8, 4724),
    ("roni_dark/2023_08_06_40ms/31to60",    32,   8, 6211),
    ("roni_dark/2023_08_06_40ms/61to90",    63,   8, 5251),
    ("roni_dark/2023_08_06_40ms/91to116",  103,   8, 3845),
    ("roni_dark/2023_08_07_5ms/1to30",      15,   8, 4671),
    ("roni_dark/2023_08_07_5ms/31to60",     35,   8, 6211),
    ("roni_dark/2023_08_07_5ms/61to90",     69,  14, 3233),
    ("roni_dark/2023_08_07_5ms/91to102",    96,   8, 6211),
    ("roni_dark/2023_08_09_60ms/121to150", 150,   8, 6211),
    ("roni_dark/2023_08_09_60ms/61to90",    86,   8, 6211),
    ("roni_dark/2023_08_10_100ms/41to70",   52,   8, 6211),
    ("roni_dark/2023_08_10_100ms/71to100",  71,   8, 6211),
]


def free_gb(path="/cs/labs/tsevi"):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1024**3


def main():
    ok, failed, incomplete, unverified = [], [], [], []
    for exp, mn, start_ind, end_ind in TARGETS:
        exp_dir = os.path.join("inference_datasets", exp)
        movie_dir = os.path.join(exp_dir, f"mov{mn}")
        label = f"{exp}/mov{mn}"

        avail = free_gb()
        if avail < MIN_FREE_GB:
            print(f"\n!! ABORTING: only {avail:.1f} GB free (need {MIN_FREE_GB}); "
                  f"{len(TARGETS) - len(ok) - len(failed)} movie(s) not attempted",
                  flush=True)
            break

        # Idempotent: a movie already rebuilt and complete is left alone, so
        # the script can be re-run after a partial pass.
        existing = find_movie_h5(movie_dir)
        if existing:
            with h5py.File(existing, "r") as f:
                if check_build_complete(f) is None:
                    print(f"\n-- {label}: already complete, skipping", flush=True)
                    ok.append(label + " (already present)")
                    continue

        print(f"\n{'=' * 70}\n-- {label}   range {start_ind}..{end_ind} "
              f"({end_ind - start_ind + 1} frames)   [{avail:.1f} GB free]",
              flush=True)
        t0 = time.time()
        rc, n_built = build_one_movie(exp_dir, movie_dir, mn,
                                      start_ind=start_ind, end_ind=end_ind)
        t1 = time.time()
        record_timing(os.path.join(exp_dir, "pipeline_timings.csv"),
                      f"mov{mn}", "build", t0, t1, n_frames=n_built)

        if rc != 0:
            print(f"   FAILED (rc={rc}) after {t1 - t0:.0f}s", flush=True)
            failed.append(label)
            continue

        h5 = find_movie_h5(movie_dir)
        with h5py.File(h5, "r") as f:
            partial = check_build_complete(f)
        if partial is not None:
            print(f"   built but INCOMPLETE: {partial}", flush=True)
            incomplete.append(label)
            continue
        print(f"   OK  {n_built} frames in {t1 - t0:.0f}s "
              f"({os.path.getsize(h5) / 1024**2:.0f} MB)", flush=True)

        calib = find_calibration_h5(exp_dir, "multi", [(movie_dir, mn)])
        if calib:
            status, medians, info = verify_one_movie(h5, calib, 15.0)
            meds = ("[" + ", ".join(f"{m:.1f}" for m in medians) + "]"
                    if isinstance(medians, list) else str(medians))
            print(f"   verify: {status}  {meds}  {info}", flush=True)
            if status != "PASS":
                unverified.append(f"{label} ({status})")
        else:
            print("   verify: no calibration.h5 found", flush=True)
            unverified.append(f"{label} (no calib)")
        ok.append(label)

    print(f"\n{'=' * 70}\nREBUILD SUMMARY")
    print(f"  rebuilt + complete : {len(ok)}/{len(TARGETS)}")
    for x in ok:
        print(f"     {x}")
    for name, lst in (("build failed", failed), ("still incomplete", incomplete),
                      ("did not verify", unverified)):
        if lst:
            print(f"  {name}: {len(lst)}")
            for x in lst:
                print(f"     {x}")
    print(f"  free space now: {free_gb():.1f} GB")


if __name__ == "__main__":
    main()

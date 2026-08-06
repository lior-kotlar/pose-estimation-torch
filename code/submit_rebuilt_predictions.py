"""Queue predictions for the rebuilt movies, one SLURM run name per experiment.

A prediction run directory is named after the movie h5 basename
(`mov_<N>_<start>_<end>_ds_..`), which does not encode the experiment. Movie
numbers repeat across experiments -- `mov_35_8_6211_ds_3tc_7tj` exists in both
2023_08_07_10ms/31to60 and 2023_08_07_5ms/31to60 -- so predicting into a shared
run name would have one overwrite the other. Giving each experiment its own run
name keeps them apart and makes the output path state the provenance.

Each movie is checked (build complete + calibration verify) before submission;
anything that fails is reported and skipped rather than queued to fail on a GPU.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
from process_experiment import (check_build_complete, find_calibration_h5,
                                find_movie_h5, verify_one_movie)

LAUNCHER = "sbatch_files/predict_single.sh"

TARGETS = [
    ("roni_dark/2023_08_06_40ms/1to30", 14),
    ("roni_dark/2023_08_06_40ms/31to60", 32),
    ("roni_dark/2023_08_06_40ms/61to90", 63),
    ("roni_dark/2023_08_06_40ms/91to116", 103),
    ("roni_dark/2023_08_07_5ms/1to30", 15),
    ("roni_dark/2023_08_07_5ms/31to60", 35),
    ("roni_dark/2023_08_07_5ms/61to90", 69),
    ("roni_dark/2023_08_07_5ms/91to102", 96),
    ("roni_dark/2023_08_09_60ms/121to150", 150),
    ("roni_dark/2023_08_09_60ms/61to90", 86),
    ("roni_dark/2023_08_10_100ms/41to70", 52),
    ("roni_dark/2023_08_10_100ms/71to100", 71),
]


def run_name(experiment):
    """'roni_dark/2023_08_06_40ms/1to30' -> 'roni_2023_08_06_40ms_1to30'."""
    parts = experiment.split("/")
    return "_".join(["roni"] + parts[1:])


def main(dry_run=False):
    submitted, skipped = [], []
    for exp, mn in TARGETS:
        exp_dir = os.path.join("inference_datasets", exp)
        movie_dir = os.path.join(exp_dir, f"mov{mn}")
        label = f"{exp}/mov{mn}"

        h5 = find_movie_h5(movie_dir)
        if h5 is None:
            skipped.append((label, "no dataset h5 (rebuild did not produce one)"))
            continue
        with h5py.File(h5, "r") as f:
            partial = check_build_complete(f)
        if partial is not None:
            skipped.append((label, f"build INCOMPLETE {partial['n_cropzone']}/"
                                   f"{partial['n_expected']}"))
            continue
        calib = find_calibration_h5(exp_dir, "multi", [(movie_dir, mn)])
        if calib is None:
            skipped.append((label, "no calibration.h5"))
            continue
        status, medians, _info = verify_one_movie(h5, calib, 15.0)
        if status != "PASS":
            skipped.append((label, f"verify {status} {medians}"))
            continue

        name = run_name(exp)
        cmd = ["sbatch", "-J", name, LAUNCHER, movie_dir]
        if dry_run:
            print(f"  would submit: {' '.join(cmd)}", flush=True)
            submitted.append((label, name, "DRY"))
            continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            skipped.append((label, f"sbatch failed: {r.stderr.strip()}"))
            continue
        jobid = r.stdout.strip().split()[-1]
        print(f"  submitted {label} -> job {jobid}, run name '{name}'", flush=True)
        submitted.append((label, name, jobid))

    print(f"\nsubmitted {len(submitted)}/{len(TARGETS)}")
    print(f"{'movie':<42} {'run name':<30} job")
    for label, name, jobid in submitted:
        print(f"  {label:<40} {name:<30} {jobid}")
    if skipped:
        print(f"\nskipped {len(skipped)}:")
        for label, why in skipped:
            print(f"  {label}: {why}")
    print("\noutput will be at predict_output/debug_outputs/<run name>/"
          "<movie h5 basename>/ , with source.json naming the experiment")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)

# Batch pipeline: from raw sparse movies to predicted 3D pose

This guide is the start-to-finish recipe for turning a freshly-imported
experiment (raw `*_sparse.mat` movies + an easyWand calibration) into a batch of
predicted, smoothed 3D pose data plus validation plots — unattended, on the
cluster.

There are two layers:

1. **Prep** — clean, prescan, mirror-flip, build h5s, verify calibration, and
   write a manifest of good movies. CPU-only.
2. **Predict** — one GPU job-array task per good movie: 2D detection,
   2D→3D triangulation, smoothing, an mp4, and wing-angle plots.

`pipeline.sh` runs both back-to-back. You can also run each layer by hand.

> All commands are run from the repo root
> `/cs/labs/tsevi/lior.kotlar/pose-estimation-torch`. Python is always the venv
> interpreter: `.env/bin/python` (or `source .env/bin/activate` first).

---

## 0. What you need before you start

- An **input directory** of raw movies. Two layouts are auto-detected:
  - **multi-movie**: `<input_dir>/mov1/`, `mov2/`, … each holding 4
    `*_cam<N>_sparse.mat` files. (This is the normal case for an experiment.)
  - **single-movie**: `<input_dir>` itself holds the 4 `*_sparse.mat` files and
    its basename is `mov<N>`.
- The **easyWand calibration `.mat`** for that experiment. For multi-day Roni
  experiments use the END-of-experiment easyWand mat for every movie
  (see the project calibration convention).
- The **mirror camera name** for this rig: `cam1` for 2023 data, `cam5` for
  2022 data (cam5 is mirrored — both the dataset h5 and the calibration must be
  vertically flipped).
- A **predict config** JSON (e.g. `predict_configurations/config1.json`). Paths
  in it are repo-relative. You normally only edit `number of cameras` and the
  model/calibration fields; `data directory`, `general run name`, and
  `pipeline timings path` are overwritten per-task by the launcher.

---

## 1. One-shot: prep + predict in one submission (recommended)

```bash
sbatch -J <experiment_name> sbatch_files/pipeline.sh \
    <input_dir> <easywand.mat> <cam> <predict_config> [array_concurrency]
```

Example:

```bash
sbatch -J 2023_mov101to110 sbatch_files/pipeline.sh \
    inference_datasets/test/2023/101to110 \
    inference_datasets/.../10_8_23_allmovs_easyWandData.mat \
    cam1 predict_configurations/config1.json 32
```

What happens:

1. The CPU prep job runs `process_experiment.py` (clean → prescan → flip →
   build → verify → manifest), appending per-step timings to
   `<input_dir>/pipeline_timings.csv`.
2. If `manifests/good_movies_<experiment_name>.txt` ends up non-empty, the job
   submits `predict_array.sh` as a **separate** GPU array job, sized to the
   manifest and named after `-J` so every task lands under one run directory.
3. The prep job exits as soon as the array is submitted; the array runs in
   parallel on GPU nodes.

`-J <experiment_name>` becomes both the prep log name and the predict run name
(the parent output directory). The optional 5th arg caps concurrent GPU tasks
(default 32).

> The `<experiment_name>` you pass to `-J` should match the input dir's basename
> (e.g. `101to110`) so the manifest the prep step writes
> (`manifests/good_movies_<basename>.txt`) is the one `pipeline.sh` looks for.

---

## 2. Running the two layers by hand

### 2a. Prep only

```bash
# typically on a CPU sbatch (no GPU needed):
sbatch --gres=gpu:0 -J prep_101to110 sbatch_files/sbatch_configurable.sh \
    code/process_experiment.py \
    inference_datasets/test/2023/101to110 \
    --easywand inference_datasets/.../10_8_23_allmovs_easyWandData.mat \
    --cam cam1 --verify

# or directly (small inputs / debugging):
.env/bin/python code/process_experiment.py <input_dir> \
    --easywand <easywand.mat> --cam <cam> [flags]
```

Useful flags (see `--help` for the full list):

| flag | effect |
|------|--------|
| `--max-frames N` | cap each movie to N frames (quick test runs) |
| `--prescan-min-intersection N` | min all-4-cam single-fly run to keep a movie (default 500) |
| `--prescan-only` | only run the prescan, then stop |
| `--verify-only` / `--no-verify` | run only / skip the reprojection sanity check |
| `--verify-threshold PX` | flag movies whose reprojection error exceeds PX (default 15) |
| `--skip-clean / --skip-flip / --skip-build` | skip individual stages |
| `--dry-run` | print what would happen, change nothing |

Outputs of prep:
- one `mov_<n>_<start>_<end>_ds_*tc_*tj.h5` per movie (the dataset h5),
- one shared `<input_dir>/calibration.h5`,
- `manifests/good_movies_<experiment>.txt` (the good-movie list),
- `<input_dir>/process_report.txt` (prescan + verify transcript),
- `<input_dir>/pipeline_timings.csv` (per-step timings).

Inspect `process_report.txt` and the prescan/verify output before predicting.

### 2b. Predict only (movies already built)

```bash
# 1. build a manifest of movie dirs (one per line, no trailing slash),
#    on a SHARED filesystem (NOT /tmp):
mkdir -p manifests
ls -d inference_datasets/test/2023/101to110/mov* > manifests/movies_101to110.txt

# 2. submit the array:
N=$(wc -l < manifests/movies_101to110.txt)
sbatch --array=0-$((N-1))%32 -J 2023_mov101to110 \
    sbatch_files/predict_array.sh \
    manifests/movies_101to110.txt predict_configurations/config1.json
```

Each task picks its movie from the manifest line `SLURM_ARRAY_TASK_ID + 1`,
writes a per-task temp config pointing `data directory` at just that movie,
stamps `general run name` = the `-J` job name, and runs
`code/prediction_code_lior/predict.py`.

To re-run only failed tasks: `sbatch --array=12,45,108%16 ...`.

---

## 3. Where the results land

```
predict_output/<run_name>/<mov_name>/
    points_3D_smoothed_ensemble_best_method.npy   # the predicted 3D pose
    points_ensemble_smoothed_reprojected.npy      # 2D reprojections
    movie 2D and 3D.mp4                            # rendered 2D+3D animation
    <mov>_analysis_smoothed.h5                     # analysis (wing angles etc.)
    wing-angle validation PNGs
```

`<run_name>` is the `-J` job name (so all movies of one experiment share a
parent). Per-movie/per-step timings accumulate in the experiment's
`pipeline_timings.csv` (`predict`, `plot`, `total` rows joined on `mov<N>`).

---

## 4. Quick sanity checks & standalone tools

These power the pipeline but are runnable on their own:

```bash
# Is a movie worth processing? (longest all-4-cam single-fly run)
.env/bin/python code/scan_sparse_movies.py <movie_dir>

# Reprojection-error check of a built h5 against calibration.h5
.env/bin/python code/verify_calibration.py <movie.h5> <calibration.h5>

# Flip the mirror cam's sparse mat in place (single or batch)
.env/bin/python code/flip_sparse_cam_mat.py <movies_dir> --cam cam1 --dry-run

# Wing-angle plot from an analysis h5 (or a dir of them)
.env/bin/python code/plot_wing_angles.py <dir>

# Shrink an h5 to its first N frames (fast iteration)
.env/bin/python code/truncate_h5_movie.py <movie.h5> 1500
```

MATLAB build/calibration scripts (driven by `code/build_experiment.sh`, but
overridable from the CLI) live in `matlab/` and addpath into the vendored
`micro-flight-lab-master/` for `HullReconstruction`.

---

## 5. Typical end-to-end run, condensed

```bash
# new experiment just imported to inference_datasets/test/2023/101to110,
# 2023 rig (mirror cam = cam1), end-of-experiment easyWand mat in hand:

sbatch -J 101to110 sbatch_files/pipeline.sh \
    inference_datasets/test/2023/101to110 \
    inference_datasets/test/2023/10_8_23_allmovs_easyWandData.mat \
    cam1 predict_configurations/config1.json 32

# watch it:
squeue -u $USER
tail -f logs/101to110_*.out                 # prep job
tail -f logs/101to110_*_*.out               # predict array tasks

# when done, results are under predict_output/101to110/mov*/
```

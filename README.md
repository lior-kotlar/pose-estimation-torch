# pose-estimation-torch

A PyTorch pipeline for **3D pose estimation of flying insects** from
multi-camera high-speed video. Given four synchronized camera views of a fly,
it detects 2D landmarks (wing and body points) with CNNs, triangulates them
into 3D using camera calibration, smooths the trajectory, and extracts flight
kinematics (wing angles, body orientation, etc.).

> **Environment note:** this project runs on the HUJI SLURM cluster. All paths
> are relative to the repo root
> `/cs/labs/tsevi/lior.kotlar/pose-estimation-torch`, and Python is **always**
> the project virtualenv interpreter `.env/bin/python` — never the system
> `python3`.

---

## Table of contents

1. [How it works (the big picture)](#1-how-it-works-the-big-picture)
2. [Repository layout](#2-repository-layout)
3. [Setup](#3-setup)
4. [Key concepts & data formats](#4-key-concepts--data-formats)
5. [Preparing new movies for prediction](#5-preparing-new-movies-for-prediction)
6. [Prediction](#6-prediction)
7. [Training](#7-training)
8. [Standalone / utility tools](#8-standalone--utility-tools)
9. [Tips & troubleshooting](#9-tips--troubleshooting)

---

## 1. How it works (the big picture)

An experiment consists of many short **movies**, each recorded by **4 cameras**.
Turning raw footage into 3D flight data has three stages:

```
 raw *_sparse.mat  ──►  built .h5 movie + calibration.h5  ──►  3D pose + kinematics
   (per camera)          (prep / MATLAB build)                 (predict / GPU)
```

1. **Prepare** — clean, scan, mirror-flip, and build each movie into a single
   `.h5`, plus one shared `calibration.h5` from the camera calibration.
   → [Section 5](#5-preparing-new-movies-for-prediction)
2. **Predict** — an **ensemble** of trained 2D-landmark CNNs runs on each movie;
   the 2D points are triangulated into 3D, smoothed, rendered to an mp4, and
   analysed into wing/body kinematics. → [Section 6](#6-prediction)
3. **Train** — you train the 2D-landmark CNNs on a labelled dataset `.h5`.
   The ensemble that Predict uses is just a set of these trained models.
   → [Section 7](#7-training)

Everything heavy runs as SLURM batch jobs (`sbatch`), so you rarely run Python
directly except for quick tests.

---

## 2. Repository layout

| Path | What it is |
|------|-----------|
| `code/training_code/` | Training: `train.py`, `Network.py`, `Preprocessor.py`, `Datasets.py`, `Losses.py`, `Callbacks.py`, `constants.py` |
| `code/prediction_code_lior/` | Prediction: `predict.py`, `Predictor.py`, `Triangulator.py`, `From_2D_to_3D.py`, `extract_flight_data.py`, `Visualizer.py` |
| `code/process_experiment.py` | One-command movie prep (clean → prescan → flip → build → verify → manifest) |
| `code/build_experiment.sh` | Drives the MATLAB scripts that build `.h5` movies + `calibration.h5` |
| `code/*.py` (misc) | Standalone tools — see [Section 8](#8-standalone--utility-tools) |
| `train_configurations/*.json` | Training configs (hyperparameters, model type, data path) |
| `predict_configurations/*.json` | Prediction configs + the ensemble model bank |
| `sbatch_files/*.sh` | SLURM job scripts (train, predict, pipeline) |
| `matlab/` | MATLAB scripts for building `.h5` movies, calibration, and datasets |
| `micro-flight-lab-master/` | Vendored lab MATLAB utilities (Hull reconstruction, Cine→sparse, …) |
| `train_output/`, `predict_output/` | Run outputs (gitignored) |
| `logs/` | SLURM stdout/stderr (`%x_%J.out/.err`, gitignored) |
| `PIPELINE.md` | Deep-dive on the batch prep+predict pipeline |
| `requirements.txt` | Python dependencies (pinned) |

---

## 3. Setup

### 3.1 Python environment

The virtualenv lives at `.env/` (gitignored, so it is **not** in a fresh
clone). Create it once:

```bash
cd /cs/labs/tsevi/lior.kotlar/pose-estimation-torch
python3 -m venv .env
.env/bin/pip install --upgrade pip
.env/bin/pip install -r requirements.txt
```

Key dependencies: `torch` 2.8 (CUDA 12.8), `ultralytics` (YOLO wing detector),
`h5py`, `numpy`, `scipy`, `scikit-image`, `opencv-python`, `moviepy`,
`matplotlib`. A CUDA GPU is required for training and for real prediction runs
(CPU works only for tiny debug tests).

Run Python either with the explicit interpreter or by activating the venv:

```bash
.env/bin/python code/...            # explicit (used everywhere in docs)
# or
source .env/bin/activate && python code/...
```

### 3.2 The cluster (SLURM)

Jobs are submitted with `sbatch` and monitored with `squeue`:

```bash
# submit a job, giving it a name (-J):
sbatch -J my_job sbatch_files/<script>.sh [args...]

# watch your jobs:
squeue -u $USER -o "%.9i %.9P %.20j %.13u %.2t %.8M %R"

# follow a job's log (name comes from -J):
tail -f logs/my_job_*.out
```

The GPU sbatch scripts default to an **L40S** GPU on the `salmon` partition.
Override the GPU/partition at submit time when needed, e.g.
`sbatch -p goldfish --gres=gpu:h200:1 ...`.

### 3.3 MATLAB

The **build** step of movie prep calls MATLAB (`matlab -batch ...`) to convert
sparse `.mat` movies into `.h5` and to build `calibration.h5`. Make sure
`matlab` is on your `PATH` (or set `MATLAB_BIN`). Everything *after* the build
step is pure Python.

---

## 4. Key concepts & data formats

**Movie `.h5`** — one file per movie, produced by the build step. Named
`mov_<n>_<start>_<end>_ds_3tc_7tj.h5`. Holds the 4-camera cropped image boxes
(`box`), crop offsets (`cropzone`), and metadata. This is the unit that
Predict consumes.

**`calibration.h5`** — one per experiment, built from an **easyWand** `.mat`.
Encodes the DLT parameters that map 3D world points to each camera's 2D image,
enabling triangulation and reprojection.

**Training dataset `.h5`** — a labelled set (image boxes + ground-truth
confidence maps for each joint), produced by the MATLAB dataset scripts in
`matlab/`. This is what Training consumes (config field `data path`).

**Model types** (`code/training_code/constants.py`) — how the CNN is fed and
what it predicts:

| Model type | Meaning |
|------------|---------|
| `MODEL_PER_CAM_PER_WING` | one wing at a time, one camera at a time (+ body points). The workhorse. |
| `MODEL_PER_CAM_PER_WING_UNET` | same I/O, but a true U-Net (skip connections + GroupNorm) |
| `ALL_CAMS_PER_WING` | predict one wing using all 4 cameras jointly |
| `ALL_CAMS_ALL_WINGS` | predict all points using all cameras jointly |
| `HEAD_TAIL_*` | body head/tail–only variants |

> **Torch serving note:** at *prediction* time only the served types
> `PER_WING_PER_CAM` and `PER_WING_ALL_CAMS` are wired up in the torch path.
> A model trained as `MODEL_PER_CAM_PER_WING` / `..._UNET` is **served** as
> `PER_WING_PER_CAM` (see the ensemble bank in
> `predict_configurations/config_bank.json`).

**Ensemble** — Predict does not use a single model; it runs several trained
models and combines their 2D predictions before triangulation for robustness.
The members are declared in `config_bank.json` and selected in
`specified_configs.json`. Keep the ensemble at roughly **4–6 members**.

**Mirror camera** — one camera in each rig is physically mirrored and must be
vertically flipped in **both** the movie data and the calibration:
`cam1` for **2023** rigs, `cam5` for **2022** rigs.

---

## 5. Preparing new movies for prediction

This turns a freshly-imported experiment (raw `*_sparse.mat` movies + an
easyWand calibration) into built `.h5` movies ready for Predict.

> Full detail and every flag live in **[PIPELINE.md](PIPELINE.md)**. This is the
> quick version.

### 5.1 What you need

- An **input directory** of raw movies, in one of two auto-detected layouts:
  - **multi-movie** (normal): `<input_dir>/mov1/`, `mov2/`, … each holding 4
    `*_cam<N>_sparse.mat` files.
  - **single-movie**: `<input_dir>` itself holds the 4 `*_sparse.mat` files and
    its basename is `mov<N>`.
- The **easyWand calibration `.mat`** for the experiment. For multi-day Roni
  experiments use the **end-of-experiment** easyWand mat for every movie.
- The **mirror cam name**: `cam1` (2023) or `cam5` (2022).
- A **predict config** (e.g. `predict_configurations/config1.json`).

### 5.2 One command: prep + predict together (recommended)

```bash
sbatch -J <experiment_name> sbatch_files/pipeline.sh \
    <input_dir> <easywand.mat> <cam> <predict_config> [array_concurrency]
```

Example:

```bash
sbatch -J 101to110 sbatch_files/pipeline.sh \
    inference_datasets/test/2023/101to110 \
    inference_datasets/test/2023/10_8_23_allmovs_easyWandData.mat \
    cam1 predict_configurations/config1.json 32
```

`pipeline.sh` runs the CPU prep (`process_experiment.py`), then — if any movies
pass the quality checks — automatically submits a **GPU job array** that
predicts every good movie in parallel. `-J <experiment_name>` names both the
prep log and the output run directory. Make the name match the input dir's
basename so the manifest is found.

### 5.3 Prep only (no prediction yet)

```bash
# on a CPU node (no GPU needed):
sbatch --gres=gpu:0 -J prep_101to110 sbatch_files/sbatch_configurable.sh \
    code/process_experiment.py \
    inference_datasets/test/2023/101to110 \
    --easywand inference_datasets/test/2023/10_8_23_allmovs_easyWandData.mat \
    --cam cam1

# or directly for small inputs / debugging:
.env/bin/python code/process_experiment.py <input_dir> \
    --easywand <easywand.mat> --cam <cam> [flags]
```

What it does, in order: **clean** (remove leftover project files) → **prescan**
(drop movies where the fly isn't visible in all 4 cams long enough) → **flip**
the mirror cam → **build** the `.h5` movies + `calibration.h5` (MATLAB) →
**verify** (reprojection-error sanity check) → write
`manifests/good_movies_<experiment>.txt`.

Useful flags (`--help` for all):

| flag | effect |
|------|--------|
| `--max-frames N` | cap each movie to N frames (quick test builds) |
| `--prescan-min-intersection N` | min all-4-cam single-fly run to keep a movie (default 500) |
| `--verify-only` / `--no-verify` | run only / skip the reprojection check |
| `--verify-threshold PX` | flag movies whose reprojection error exceeds PX (default 15) |
| `--skip-clean` / `--skip-flip` / `--skip-build` | skip individual stages |
| `--dry-run` | print what would happen, change nothing |

**Always review** `<input_dir>/process_report.txt` (prescan + verify transcript)
before predicting.

---

## 6. Prediction

Prediction runs the ensemble on each movie `.h5`, triangulates to 3D, smooths,
renders an mp4, and extracts kinematics.

### 6.1 The predict config

`predict_configurations/config1.json` — the fields you normally touch:

| field | meaning |
|-------|---------|
| `data directory` | folder holding the movie `.h5` files (an experiment dir, or a single `mov*` dir). The predictor walks it recursively for `mov*.h5`. |
| `output directory` | where results are written (`predict_output/...`) |
| `calibration path` | the experiment's `calibration.h5` |
| `wings detector path` | YOLO weights for wing segmentation (`wings_detetction/yolo_weights_6_1_24.pt`) |
| `config bank path` | the ensemble bank (`config_bank.json`) |
| `specified configs path` | which bank entries form the ensemble (`specified_configs.json`) |
| `number of cameras` | usually `4` |
| `IMAGE HEIGHT` / `IMAGE WIDTH` | full-frame size (800 × 1280) |

**Choosing the ensemble:** `specified_configs.json` lists the member names (e.g.
`["config1", "config2"]`); each name must exist in `config_bank.json`, where it
points at a trained model's `best_model.pt` and its served model type. To add a
member, add its block to `config_bank.json` and its name to
`specified_configs.json`.

### 6.2 Run on one experiment directory (simple path)

Edit `data directory` and `calibration path` in the config, then:

```bash
# submit on a GPU node:
sbatch -J predict_myexp sbatch_files/sbatch_configurable.sh \
    code/prediction_code_lior/predict.py predict_configurations/config1.json

# or directly (small/debug):
.env/bin/python code/prediction_code_lior/predict.py predict_configurations/config1.json
```

### 6.3 Run one movie per task in parallel (batch path)

When you already have built movies and want throughput, use the job array. This
is what `pipeline.sh` submits automatically, but you can drive it by hand:

```bash
# 1. list movie dirs into a manifest ON A SHARED FILESYSTEM (not /tmp):
mkdir -p manifests
ls -d inference_datasets/test/2023/101to110/mov* > manifests/movies_101to110.txt

# 2. submit the array (%32 caps concurrent tasks):
N=$(wc -l < manifests/movies_101to110.txt)
sbatch --array=0-$((N-1))%32 -J 101to110 \
    sbatch_files/predict_array.sh \
    manifests/movies_101to110.txt predict_configurations/config1.json
```

Each task predicts one movie, derives `calibration.h5` from the movie's parent
dir, and stamps the run name from `-J`. Re-run only failed tasks with e.g.
`--array=12,45,108%16`.

### 6.4 Where results land

```
predict_output/<run_name>/<mov_name>/
    points_3D_smoothed_ensemble_best_method.npy   # the predicted 3D pose
    points_ensemble_smoothed_reprojected.npy      # 2D reprojections
    movie 2D and 3D.mp4                            # rendered 2D+3D animation
    <mov>_analysis_smoothed.h5                     # kinematics (wing angles, …)
    <wing-angle validation PNGs>
```

`<run_name>` is the `-J` job name, so all movies of one experiment share a
parent directory.

---

## 7. Training

Training fits the 2D-landmark CNN on a labelled dataset `.h5` and writes a
timestamped run directory containing the best model (`best_model.pt`),
checkpoints, history, and a copy of the training code + config.

### 7.1 The training config

`train_configurations/config1.json` — the fields you typically set:

| field | meaning |
|-------|---------|
| `model type` | architecture / feeding scheme (see [Section 4](#4-key-concepts--data-formats)) |
| `data path` | labelled training dataset `.h5` |
| `base output directory` | where the run folder is created (`train_output/...`) |
| `epochs` | number of epochs |
| `batch size` | mini-batch size |
| `loss function` | `MSE`, `KL`, `softargmax`, or `JSD` |
| `learning rate` | initial LR (decayed by cosine annealing over the run) |
| `number of base filters`, `number of encoder decoder blocks`, `convolution kernel size`, `dilation rate`, `dropout ratio` | network shape |
| augmentation block (`rotation range`, `zoom range`, `horizontal/vertical flip`, `xy shift`, …) | data augmentation |
| `run tag` | optional short label appended to the run folder name (distinguishes variants that share a model type, e.g. `JSD`) |

The run folder is auto-named `<model_type>_<run_tag>_<date>` (e.g.
`MODEL_PER_CAM_PER_WING_JSD_Jun 30`).

### 7.2 Run a training job

```bash
# default GPU (L40S / salmon); CONFIG defaults to train_configurations/config1.json:
sbatch -J train_run sbatch_files/sbatch_job.sh train_configurations/config1.json

# or via the generic launcher (lets you override gres/partition):
sbatch -J train_jsd -p salmon --gres=gpu:l40s:1 sbatch_files/sbatch_configurable.sh \
    code/training_code/train.py train_configurations/config_per_cam_jsd.json
```

Watch it with `tail -f logs/train_run_*.out`.

### 7.3 Resuming from a checkpoint

Point the config at the checkpoint and its original run directory (see
`train_configurations/config1_resume.json`):

```json
"training checkpoint file path": ".../<run>/weights/model_epoch_49.pth",
"resume training directory": ".../<run>"
```

The resumed config must match the original on the structural fields (model type,
batch size, val fraction, kernel size, filters, blocks, loss) — the trainer
enforces this and reuses the saved train/val split for reproducibility.

### 7.4 Training models for the ensemble

The included `config_per_cam_*.json` variants exist to build a **diverse**
prediction ensemble (different loss functions and architectures over the same
`MODEL_PER_CAM_PER_WING` I/O), e.g.:

- `config_per_cam_jsd.json` — JSD loss (loss-function diversity)
- `config_per_cam_dil3.json` — different filters/dilation
- `config_per_cam_unet.json` — true U-Net variant

After training each, register its `best_model.pt` as a new member in
`config_bank.json` and add it to `specified_configs.json` — see
[Section 6.1](#61-the-predict-config).

---

## 8. Standalone / utility tools

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

# Build h5 movies + calibration.h5 directly via MATLAB (used by prep):
code/build_experiment.sh <input_dir> <easywand.mat> [--max-frames N]
```

Other analysis helpers live in `code/` (`data_analysis.py`, `comparison.py`,
`collect_analysis_h5.py`, `pipeline_timing.py`).

---

## 9. Tips & troubleshooting

- **Always use `.env/bin/python`**, not the system `python3`.
- **Manifests must be on a shared filesystem** (project dir or `$HOME`), never
  `/tmp` — compute nodes have their own `/tmp` and won't see login-node files.
- **Mirror cam:** `cam1` for 2023 data, `cam5` for 2022 data. Both the movie
  data *and* the calibration must be vertically flipped. The prep pipeline
  handles the data flip; the calibration flip is handled in the MATLAB build.
- **Calibration for multi-day Roni experiments:** use the **end-of-experiment**
  easyWand `.mat` for *all* movies, regardless of each movie's recording date.
- **A movie fails verification (high reprojection error):** the calibration
  doesn't match the data. Known-good movies reproject at ~2–8 px; known-broken
  ones at 100–400 px. Check you're using the right easyWand mat.
- **GPU queue is slow:** the L40S salmon nodes are usually free; override to
  another partition/GPU only if you need more memory or throughput.
- **Reviewing a run:** check `logs/<jobname>_*.out` / `.err`, the per-movie
  `process_report.txt`, and `pipeline_timings.csv` for per-step timings.

For the full batch-pipeline reference (every flag, every output file, timing
ledger semantics), see **[PIPELINE.md](PIPELINE.md)**.

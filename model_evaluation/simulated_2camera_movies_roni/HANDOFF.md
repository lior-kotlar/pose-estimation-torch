# Handoff: the 2-camera integration session → the main session (2026-10-01)

> **Note (2026-10-02):** this handoff is kept as a record. Its paths predate the move to `model_evaluation/`: the reports it names are now in `model_evaluation/simulated_2camera_movies_roni/` and `model_evaluation/four_camera_ensembles_roni/`, and the predictions, cut movies and helper scripts it lists were deleted once the ensembles were chosen (see `model_evaluation/README.md`).

This session brought 2-camera models and 2-camera movies into the pipeline. It also
tested them on 4 Roni movies. This file is all you need to:

- choose the ensemble for each kind of movie;
- reverse one decision if the user agrees;
- commit what is left;
- close the sprint.

The user wants the two sessions merged into yours from now on.

---

## 1. The decision to revisit first

**Your session decided:** a 4-camera movie is predicted only by the 4-camera ensemble
(4-camera + per-camera members), never by the 2-camera members.

**Where it is implemented:** `code/utils.py` `model_accepts_movie` (uncommitted). It now
requires `num_cams == model's "num cameras"` instead of `>=`. Docs follow it in:

- `code/prediction_code_lior/Predictor.py` (docstrings);
- `prediction_models/README.md`;
- `PIPELINE.md` §2c, first paragraph (the hunk near line 413).

**This session's evidence argues for reversing it.** On the 4 Roni movies the 2-camera
members are what makes the new model set better than the deployed one (§3.2):

- With them: the new set is the steadiest of all five ensembles tested.
- Without them: the new set is *worse* than the old deployed set.
- At an equal member count (4 each), including two 2-camera members still wins.
- On the 41 held-out labelled test frames (4-camera setting), 2CAM_JSD is as accurate as
  the best 4-camera model: 0.091 vs 0.089 mm (`comparison_data/heldout_uniform/report.md`).

**Cost:** a 4-camera movie with 2-camera members took 0.9–3.3 h per movie on the pilot, vs
0.85–3.1 h for the deployed set. That is about the same. Each 2-camera member runs 3 times
(once per bottom + side pair) and the selector sees more members.

**If the user reverses it:**

1. Put back `>=` in `model_accepts_movie`, and the docstrings/README/PIPELINE text that
   say so. Commit 22653ea and this session's commit f964026 hold the old wording.
2. Give each kind of movie its own member list (§2). Today one folder serves every rig,
   so a model is either in every ensemble it fits or in none. The pilot says the 2-camera
   ensemble should be 3 members but the 4-camera one more.
3. Proposed mechanism:
   - an optional `model.json` field, e.g. `"movie cameras": [2, 4]`;
   - `model_accepts_movie` honours it, and its absence means today's rule;
   - `register_prediction_model.py --movie-cameras 2 4`;
   - one line in `prediction_models/README.md`.

   All of this is in `utils.py`, which your session owns, so it is yours to write.

---

## 2. Recommended ensembles per kind of movie

"Tested" means predicted on the 4 Roni movies and scored. The member names are the
staged candidates in `prediction_models_candidates/` (Sep 28 uniform-round runs).

| movie kind | recommendation | evidence |
|---|---|---|
| **4-camera** | 4CAM DIL3, 4CAM MAXFUSION, 2CAM DIL3, 2CAM JSD, 2CAM MAXFUSION, per-cam SMALL_DIL3, per-cam UNET (the tested 7) | steadiest of 5 ensembles; beats deployed on every movie (§3.2) |
| **2-camera** | 2CAM DIL3, 2CAM JSD, per-cam SMALL_DIL3 (3) | the same accuracy as the 5-member ensemble within noise (§3.3) |
| **3-camera (Tsory)** | 3CAM models + per-cam, as your session plans | not tested here. Held-out side-triad: 3CAM_DIL3 0.091, 3CAM_MAXFUSION 0.095, 3CAM_BASE 0.101, per-cam UNET 0.111 mm. 2-camera members never run there (no bottom camera) |

Notes on the 4-camera list:

- A smaller set (M ≈ 5, per the old keep-M-small guideline) is untested; test it before
  choosing it. `comparison_data/sim_roni/rebuild_ensemble.py` rebuilds any subset of the 7
  from their saved points in `predict_output/sim_roni_all_cams/` on CPU in ~25 min; no
  GPU is needed. Then run `compare_4cam_ensembles.py` (§5).
- Per-cam UNET is the weakest single model on 2-camera movies, but it was part of the
  winning 4-camera seven, and its value there was not isolated.

**Not tested here:** your four Sep 30 runs (3CAM_JSD, 4CAM_JSD, PER_CAM_DIL3,
PER_CAM_JSD_DIL3). Score them on the held-out frames. Any that join the 4- or 2-camera
ensemble need a GPU re-prediction of the pilot's movies before they can be compared there.
The cut movies and manifests are reusable as they are (§5).

**Before replacing `prediction_models/`:**

- **Check more movies.** The pilot is 4 movies from one experiment. Old vs new should be
  compared on more 4-camera movies, by the rule that no movie gets worse.
- **Get the data.** Only these 4 Roni movies and Shalev are on the cluster, and the user
  calls Shalev "problematic". More Roni 4-camera data must come up from the PC.
- **Then deploy** under indicative names. The ~570 delivered movies hold per-member
  subfolders under the old names.

---

## 3. What was tested, and the results

### 3.1 Material and references

- **Movies:** `inference_datasets/roni/` mov1, mov5, mov12, mov13: 16 kHz, 738–2684 frames,
  bottom camera = index 0.
  - All four pass verify against its `calibration.h5` (2.4–5.9 px). mov2 FAILS
    (64–222 px), and mov3 was never built.
  - There is no `process_report`, and these movies were built before prescan sidecars
    existed.
- **The cut:** `code/make_camera_subset_movies.py` cut each movie into 4 subsets over the
  same frames:
  - the pairs `cam1_cam2`, `cam1_cam3`, `cam1_cam4`;
  - `all_cams`.

  In all 4 movies the shared window is the whole built movie.
- **Model sets:**
  - **old/deployed** = `prediction_models/` via `config1.json` (7 run on 4-camera movies:
    Dec 18 + Jul 2/5 models, trained on all 201 frames);
  - **new/candidates** = 7 Sep 28 runs staged in `prediction_models_candidates/` via
    `predict_configurations/config_candidates.json`.

### 3.2 Question 1: 4-camera ensemble composition

There is no ground truth on whole movies, so these are self-consistency scores, means
over the 4 movies; lower is better. Rigidity is the selector's own objective, so it is
necessary, not sufficient.

| ensemble | rigidity µm | jitter µm | raw→smooth µm | body-length spread mm |
|---|---|---|---|---|
| old/deployed (7) | 27.7 | 17.6 | 34.6 | 0.020 |
| **new with 2-cam members (7)** | **25.8** | **17.3** | **31.6** | **0.018** |
| new without 2-cam members (4) | 30.2 | 17.8 | 36.0 | 0.020 |
| new 2× 2CAM + 2× 4CAM (4) | 27.5 | 17.6 | 33.0 | 0.024 |
| new 4CAM only (2) | 38.4 | 19.5 | 47.9 | 0.048 |

- **Per movie:** new with 2-cam beats the deployed set on rigidity in all 4 movies.
- **Distance between them:** old vs new-with-2-cam land 0.041 mm apart (median), 0.108 mm
  at p95.
- **Member usage:** in the new 7-member ensemble the selector picked the 2-cam members on
  41–54% of frames, the 4CAM members on 15–26%.
- **Report:** `comparison_data/sim_roni/four_camera_ensembles.md`.

### 3.3 Question 2: the 2-camera ensemble, with the old/deployed 4-camera prediction as ground truth

The old/deployed ensemble shares no weights with the new models, so it is the fair
reference. The candidate `all_cams` run flatters the pairs, because the same models ran on
the same images. A 4CAM-only reference is too noisy.

**3D points:** median 0.069–0.071 mm, p95 0.19–0.21 mm, 0.2–0.7% of points beyond 0.5 mm.
There were no wing-label swaps and no NaN frames.

**Angles** (median / p95):

| angle | difference |
|---|---|
| wing φ | 1.6–2.0° / 6–9° |
| wing θ | 2.2–2.6° / 8–11° |
| wing ψ | 4.5–5.0° / 16–18° |
| body yaw | 0.6–1.0° / 5–7.5° |
| body pitch | 0.7–1.9° / 4–6.5° |
| body roll | 0.3–0.6° / 1.4–4° |

**Each member alone, before the analysis step** (median / p95, mm):

| member | distance |
|---|---|
| ensemble | 0.080–0.082 / 0.13–0.15 |
| 2CAM DIL3 | 0.087–0.091 / ~0.17 |
| 2CAM JSD | 0.090–0.094 / 0.15–0.18 |
| 2CAM MAXFUSION | ~0.10 |
| per-cam SMALL_DIL3 | ~0.11 |
| per-cam UNET | 0.13–0.14 / ~0.30 |

**Trimmed ensemble** (2CAM DIL3 + 2CAM JSD + per-cam SMALL_DIL3): 0.070–0.074 /
0.195–0.220, against 0.069–0.071 / 0.190–0.212 for all 5. Equal within noise.

**Body failure mode:** in 2 of the 12 runs the body axis was wrong for the whole flight,
with the fly short and tilted 4–8°, while the other pairs got the same movie right:

- mov13 with cam1_cam2: 2.27 mm vs 2.62 mm, −13%;
- mov1 with cam1_cam4: 2.26 mm vs 2.47 mm, −9%.

The hypothesis "the axis near the cameras' epipolar plane" was tested and is **not**
supported, so the failure is specific to the flight. **`code/check_body_length.py` flags
exactly these 2 and nothing else** (threshold 6%).

**Steadiness:** the pipeline's own rigidity score is 1.4–2× worse for the 2-camera runs
(33–63 µm) than for the 4-camera runs (23–30 µm).

**No best pair:** each pair has one bad movie.

**Reports:**

- `comparison_data/sim_roni/vs_deployed/report.md` (main);
- `…/vs_deployed/trimmed_3_members/`;
- figures `fig_pairs_vs_reference.png` and `fig_body_vs_axis_angle.png`.

### 3.4 2-camera prep and its limits

These matter for a real 2-camera rig.

- **Declared, never detected.** A 2-camera experiment is declared:
  `process_experiment.py --num-cams 2 --bottom-cam N`.
  - Undeclared 2-mat folders stay "incomplete" (Tsory ex210826 has 11).
  - `--bottom-cam auto` is refused before any slow step.
- **Real 2-camera prep equals the cut, bit for bit:** box, crop positions and per-camera
  calibration (smoke test on copied Shalev mats + a cut-down 2-camera easyWand, since
  deleted).
- **Verify on 2 cameras** uses the both-views residual, threshold 6 px. Measured: 0.6–2.5 px
  with the right calibration, ≥12.4 px with another rig's.
- **The mirror check cannot see a flipped side camera** with two views (2.9 px vs 1.4 px),
  so it reports INCONCLUSIVE.
  - A flipped bottom camera, the mirrored one, shows on some pairs (143–219 px) but not
    all (4.9 px).
  - **The mirror camera must be known for the rig.**
- **A 2-camera easyWand tilts the lab +z by 27°**, the bisector of the two views. A real
  rig needs its own "up" before body angles can be trusted. The simulation avoids this by
  keeping the 4-camera lab frame.

---

## 4. Git state and what is left to commit

**Committed:** `f964026`, on main. It holds:

- 2-camera prep: `scan_sparse_movies.py`, `process_experiment.py`, `find_mirror_cam.py`;
- the two new tools;
- `config_candidates.json`;
- `.gitignore` (`prediction_models_candidates/` ignored);
- docs.

**Uncommitted and this session's own** (all hunks):

- `code/check_body_length.py` (new);
- `code/compare_camera_subsets.py` (all changes since f964026): `--reference-run`, member
  scores, body length, the axis-angle analysis, a legend fix;
- `PIPELINE.md`: the two LATER hunks only, the §2c paragraph from "Which 4-camera
  prediction stands in for the truth" through "What the Roni pilot showed", and the
  `check_body_length.py` lines in §4. **The first hunk (§2c opening, near line 413) is
  your session's.**

**Uncommitted and your session's:** `README.md`, `code/utils.py`,
`code/prediction_code_lior/Predictor.py`, `prediction_models/README.md`,
`code/evaluate_on_heldout.py`, `code/training_code/*`, `sbatch_files/sbatch_configurable.sh`,
`train_configurations/*`.

**Index trap:** your session has a staged deletion of
`train_configurations/config_all_cams_2cam_dil3_long.json`. `git commit` takes the whole
index, not just the paths added. It slipped into this session's first commit attempt;
that commit was redone and the staged deletion was restored. Stage explicitly, and check
`git diff --cached --stat` before committing.

**Commit style** (user's rule): few numbered bullets, short and high-level, no co-author
line, on main, only when the user asks.

---

## 5. Where everything is

- **Cut movies:** `inference_datasets/simulated/roni/{all_cams,cam1_cam2,cam1_cam3,cam1_cam4}/mov{1,5,12,13}/`.
  - Each holds symlinks to the source mats.
  - Prep refuses these folders, because they hold `derived_from.json`.
- **Manifests** (gitignored): `manifests/good_movies_roni.txt` (the 4 verify-passed movies),
  `manifests/sim_roni_<subset>.txt`.
- **Predictions:**
  - `predict_output/sim_roni_{all_cams,cam1_cam2,cam1_cam3,cam1_cam4}` (candidates);
  - `predict_output/sim_roni_all_cams_deployed` (old models).
  - All are evaluation only, never delivered.
- **Ensemble variants rebuilt on CPU from saved member points:**
  `comparison_data/sim_roni/ensemble_variants/{no_2cam_members,four_with_2cam,pairs_trimmed}/`
  and `comparison_data/sim_roni/independent_reference/` (4CAM-only).
- **Pilot scaffolding** (untracked, keep or delete):
  - `comparison_data/sim_roni/rebuild_ensemble.py` (rebuild any member subset);
  - `compare_4cam_ensembles.py`;
  - `independent_reference/make_reference.py`.
- **Re-running the pilot with new members:**
  1. Register them into `prediction_models_candidates/` (`register_prediction_model.py
     --prediction-models-dir prediction_models_candidates`; train type
     MODEL_PER_CAM_PER_WING_UNET needs `--type PER_WING_PER_CAM`).
  2. Submit `sbatch -J sim_roni_<subset>_v2 --array=0-3%4 -p catfish,salmon --gres=gpu:1
     --mem=96g --cpus-per-task=12 sbatch_files/predict_array.sh manifests/sim_roni_<subset>.txt
     predict_configurations/config_candidates.json`, using a new run name so the old run is
     kept.
  3. Score with `.env/bin/python code/compare_camera_subsets.py inference_datasets/simulated/roni
     --reference-run predict_output/sim_roni_all_cams_deployed`. Its `--predict-output`
     expects `sim_roni_<subset>` run names, so point it at a folder of symlinks if the run
     names differ.
- **Memory:** `bottom-camera-geometry.md` carries the durable facts and numbers above.

---

## 6. Caveats

- **Small sample:** 4 movies from one experiment.
- **No labelled truth:** every movie-level comparison is against a 4-camera prediction. On
  the hardest movie (mov13), even two 4-camera ensembles disagree by up to ~5° of body yaw.
- **Steadiness isn't accuracy:** the steadiness scores are not truth. The held-out labelled
  frames are the only real accuracy numbers.
- **What would settle it:** the planned annotation round, if it labels frames inside whole
  movies, gives true accuracy for both rigs and for the ensembles.

# prediction_models/

Single source of truth for the **prediction ensemble**. The prediction pipeline
auto-discovers every model folder here — there is no central list to edit.

## Layout

```
prediction_models/
    <name>/
        best_model.pt      # torchscript weights (git-ignored — artifact)
        model.json         # metadata (git-tracked)
    ...
```

`model.json`:

```json
{
    "model type": "PER_WING_ALL_CAMS",   // or "PER_WING_PER_CAM" (only these work with torch)
    "num cameras": 4,                     // 4 | 3 | 2 | "any" — see below
    "bottom camera": "required",          // only on the 2-camera bottom + side models
    "movie cameras": [2, 4],              // optional: the movies it is chosen for — see below
    "enabled": true,                      // set false to bench a model without deleting it
    "predict again 3D consistency": 0,
    "use reprojected masks": 0,
    "source": "train_output/debug_outputs/ALL_CAMS_PER_WING_DIL3_Jul 05_01"
}
```

A folder is used as an ensemble member iff it contains **both** `best_model.pt`
and `model.json` and `enabled != false`. Members are ordered by folder name
(this order is the model index in the selection report).

## `num cameras` — which movies a model can run on

The lab has two rigs: the current 4-camera one, and an older 3-camera one that
predates the fourth camera. `predict.py` reads a movie's camera count from its
box (`cropzone.shape[1]`) and keeps only the members that fit:

| value | meaning | which models |
|-------|---------|--------------|
| `4` | 4-camera movies only | every `PER_WING_ALL_CAMS` trained on 4 cameras |
| `3` | 3-camera movies only | a `PER_WING_ALL_CAMS` trained on 3 |
| `"any"` (or absent) | any camera count | every `PER_WING_PER_CAM` |
| `2` + `"bottom camera": "required"` | movies with a bottom camera: 2-camera ones, and more-camera ones when `"movie cameras"` lists them | a `PER_WING_ALL_CAMS` trained on (bottom, side) pairs |

### `"bottom camera": "required"` — the 2-camera models

These take two cameras, and the bottom camera is always in slot 0, as it was in
training. On a 2-camera (bottom + side) movie they run once. A movie with more
cameras runs them only when their `"movie cameras"` lists it; then they run
once per (bottom, side) pair:

- each side camera gets its points from its own run;
- the bottom camera's confmaps are averaged over the runs before the peak is taken.

The output has the same shape as any other member's, so triangulation and the
ensemble treat it the same way.

The movie's bottom camera comes from `calibration.h5`. Prep's `--bottom-cam`
writes it there; when it is absent, prediction works it out from the camera
positions (`find_bottom_camera` in `code/utils.py`). The old 3-camera rig is the
side triad, with no bottom camera, so these models never run on it.
`register_prediction_model.py --from <run>` adds the field when the run's config
has `"required camera": "bottom"`.

The distinction is structural, not a preference: an ALL_CAMS model fuses a
fixed number of camera streams (4 cameras = 16 input channels → 40 output
confmaps), so its weights simply do not fit a 3-camera movie. A PER_CAM model
sees one camera at a time, so it is camera-count agnostic and a 4-camera-trained
one applies unchanged to 3-camera movies.

Omitting the field means `"any"`, so a `model.json` written before this existed
keeps working. `register_prediction_model.py` fills it in automatically (from
the training config's `number of cameras`, or `--num-cams`).

A 3-camera movie with no matching ALL_CAMS member still predicts — on the
per-cam members alone. If *no* member matches, prediction stops with an error
rather than silently producing nothing.

## `movie cameras` — a different ensemble for each kind of movie

`num cameras` says which movies a model *can* run on. The optional
`"movie cameras"` says which ones it is *chosen* for: the camera counts of those
movies, e.g. `[2, 3]` for a per-camera model used on 2- and 3-camera movies but
not 4-camera ones, or `[2, 4]` for a 2-camera model used on 4-camera movies too.
So one folder holds every ensemble. Without the field a model runs on every
movie it can, except that a bottom + side model stays on 2-camera movies.
Prediction logs which members it kept for a movie and why it skipped the others.
`register_prediction_model.py --movie-cameras 2 4` writes it, and refuses a
camera count the weights cannot run on.

## Staging candidates

Models not yet deployed are registered into `prediction_models_candidates/`
(`register_prediction_model.py --prediction-models-dir prediction_models_candidates`)
and predicted with `predict_configurations/config_candidates.json`, e.g. for the
simulated 2-camera movies (PIPELINE.md 2c). That folder is git-ignored local
state; a member is deployed by registering it here.

The weights are **git-ignored**; only `model.json` (and this README) are tracked,
so the set of models and their types is versioned while the large binaries live
on the lab filesystem.

## Add a model (graduate a trained model)

```bash
# infer the type from a finished training run:
python code/register_prediction_model.py --name per_cam_dil3 \
    --from "train_output/debug_outputs/MODEL_PER_CAM_PER_WING_DIL3_Jul 02"

# or specify weights + type explicitly:
python code/register_prediction_model.py --name my_model \
    --type PER_WING_PER_CAM --weights path/to/best_model.pt
```

Training happens freely in `train_output/`; nothing there is used for prediction
until you register it here. Prediction always uses exactly the enabled folders in
this directory.

## Remove / disable a model

- Disable: set `"enabled": false` in its `model.json`.
- Remove: delete its folder.

## Run prediction

```bash
# one movie:
sbatch sbatch_files/predict_single.sh <movie_dir>

# a batch (manifest of movie dirs):
sbatch predict_array.sh <manifest> predict_configurations/config1.json
```

`config1.json` carries `"prediction models directory": "prediction_models"` and
`"max ensemble models"` (caps the selector's per-window model-subset search, which
grows ~2^M in the number of members).

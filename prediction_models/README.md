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
    "enabled": true,                      // set false to bench a model without deleting it
    "predict again 3D consistency": 0,
    "use reprojected masks": 0,
    "source": "train_output/debug_outputs/ALL_CAMS_PER_WING_DIL3_Jul 05_01"
}
```

A folder is used as an ensemble member iff it contains **both** `best_model.pt`
and `model.json` and `enabled != false`. Members are ordered by folder name
(this order is the model index in the selection report).

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

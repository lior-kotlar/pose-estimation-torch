#!/usr/bin/env python
"""
Register (graduate) a trained model into the prediction ensemble.

The prediction pipeline auto-discovers every model folder under
``prediction_models/`` (see PredictConfig.load_configurations_from_models_dir).
A model folder is self-contained:

    prediction_models/<name>/
        best_model.pt      # torchscript weights
        model.json         # {"model type": "PER_WING_ALL_CAMS", "enabled": true, ...}

This script creates such a folder from either a finished training run directory
(inferring the predict model type from its saved config) or an explicit
weights path + type.

Examples
--------
    # graduate a trained model (type inferred from the train config):
    python code/register_prediction_model.py --name per_cam_dil3 \
        --from "train_output/debug_outputs/MODEL_PER_CAM_PER_WING_DIL3_Jul 02"

    # explicit weights + type:
    python code/register_prediction_model.py --name my_model \
        --type PER_WING_PER_CAM --weights path/to/best_model.pt

    # bench it later without deleting:  edit model.json -> "enabled": false
"""
import os
import sys
import json
import glob
import shutil
import argparse

# predict model types that actually work with torchscript .pt models
VALID_PREDICT_TYPES = {"PER_WING_ALL_CAMS", "PER_WING_PER_CAM"}

# train-side model type -> predict-side model type
TRAIN_TO_PREDICT_TYPE = {
    "ALL_CAMS_PER_WING": "PER_WING_ALL_CAMS",
    "MODEL_PER_CAM_PER_WING": "PER_WING_PER_CAM",
}

WEIGHTS_FILE = "best_model.pt"
META_FILE = "model.json"


def _read_train_type(train_dir):
    """Return the predict model type inferred from a train run dir's saved config."""
    # training saves its config as a json in the run dir; find the one that
    # carries a "model type" field.
    for cfg_path in sorted(glob.glob(os.path.join(train_dir, "*.json"))):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except (ValueError, OSError):
            continue
        train_type = cfg.get("model type")
        if train_type in TRAIN_TO_PREDICT_TYPE:
            return TRAIN_TO_PREDICT_TYPE[train_type]
        if train_type in VALID_PREDICT_TYPES:  # already a predict-side type
            return train_type
    return None


def resolve_source(args):
    """Return (weights_path, model_type, source_note) from the CLI args."""
    if args.from_dir:
        weights_path = os.path.join(args.from_dir, WEIGHTS_FILE)
        if not os.path.isfile(weights_path):
            sys.exit(f"error: {WEIGHTS_FILE} not found in {args.from_dir}")
        model_type = args.type or _read_train_type(args.from_dir)
        if model_type is None:
            sys.exit("error: could not infer 'model type' from the train dir; "
                     "pass --type PER_WING_ALL_CAMS | PER_WING_PER_CAM explicitly")
        return weights_path, model_type, args.from_dir
    # explicit weights + type
    if not args.weights or not args.type:
        sys.exit("error: provide either --from <train_dir> or both --weights and --type")
    if not os.path.isfile(args.weights):
        sys.exit(f"error: weights not found: {args.weights}")
    return args.weights, args.type, os.path.dirname(args.weights)


def main():
    ap = argparse.ArgumentParser(description="Register a model into prediction_models/.")
    ap.add_argument("--name", required=True, help="folder name under the prediction models dir")
    ap.add_argument("--from", dest="from_dir", help="finished training run directory")
    ap.add_argument("--weights", help="path to best_model.pt (with --type)")
    ap.add_argument("--type", choices=sorted(VALID_PREDICT_TYPES),
                    help="predict model type (overrides / required when not using --from)")
    ap.add_argument("--prediction-models-dir", default="prediction_models",
                    help="target registry directory (default: prediction_models)")
    ap.add_argument("--disabled", action="store_true",
                    help="write the model with enabled=false (staged, not used yet)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing model folder")
    args = ap.parse_args()

    if args.type and args.type not in VALID_PREDICT_TYPES:
        sys.exit(f"error: --type must be one of {sorted(VALID_PREDICT_TYPES)}")

    weights_path, model_type, source_note = resolve_source(args)

    dest_dir = os.path.join(args.prediction_models_dir, args.name)
    if os.path.exists(dest_dir) and not args.force:
        sys.exit(f"error: {dest_dir} already exists (use --force to overwrite)")
    os.makedirs(dest_dir, exist_ok=True)

    shutil.copy2(weights_path, os.path.join(dest_dir, WEIGHTS_FILE))
    meta = {
        "model type": model_type,
        "enabled": not args.disabled,
        "predict again 3D consistency": 0,
        "use reprojected masks": 0,
        "source": source_note,
    }
    with open(os.path.join(dest_dir, META_FILE), "w") as f:
        json.dump(meta, f, indent=4)

    print(f"registered '{args.name}': type={model_type}, enabled={not args.disabled}")
    print(f"  weights <- {weights_path}")
    print(f"  wrote   -> {dest_dir}/")


if __name__ == "__main__":
    main()

# model_evaluation/

How well the pose-estimation models and ensembles do, and the evidence behind
the ensemble deployed on 2026-10-02 (the `*_v2` members in `prediction_models/`).
Only the reports are kept; everything they were computed from can be rebuilt
with the commands below.

| folder | what it answers |
|---|---|
| `heldout_test_frames/` | how accurate is each model, against labels? |
| `simulated_2camera_movies_roni/` | can the 2-camera models stand in for a 2-camera rig, on whole movies? |
| `four_camera_ensembles_roni/` | which ensemble should predict 4-camera movies? |
| `hull_comparison/` | how does the method compare with the Hull reconstruction? (older) |

## heldout_test_frames/

Every model scored on the 41 labelled frames that no model trains on (the test
frames of `training_datasets/random_trainset_201_frames_18_joints.split_v1.npz`),
separately for each camera rig: a bottom + side pair, all 4 cameras, and the
side triad (the old 3-camera rig). 2D error in pixels, 3D error in mm against
the labelled 3D points, and `COMBINED_MEDIAN`, the median of all models' 3D
points as a stand-in for the ensemble. `deployed_*` rows are the members retired
on 2026-10-02 (`prediction_models_retired/`); they trained on most of these
frames, so their scores are optimistic.

`report.md` has the tables, `summary.json` every number, and the figures one bar
chart per rig plus the error tails. Rebuilt (about 5 minutes on a GPU node) by:

```bash
sbatch -J eval_heldout -p catfish,salmon --gres=gpu:1 --mem=128g --time=2:00:00 \
    sbatch_files/sbatch_configurable.sh code/evaluate_on_heldout.py \
    --out model_evaluation/heldout_test_frames \
    --models "train_output/debug_outputs/<run>" ... prediction_models_retired/*/
```

## simulated_2camera_movies_roni/

Four Roni 4-camera movies (mov1, mov5, mov12, mov13) cut into their three
(bottom, side) pairs and predicted as 2-camera movies, then compared frame by
frame with a 4-camera prediction of the same frames (no labels exist on whole
movies). Made with the candidate models of 2026-10-01, before the final set was
chosen.

- `vs_deployed/report.md`: the main comparison, against the then-deployed
  models' 4-camera prediction, which shares no weights with the pairs.
  `vs_deployed/trimmed_3_members/` repeats it with a 3-member 2-camera ensemble.
- `report.md`: against the candidates' own 4-camera run (flatters the pairs).
- `HANDOFF.md`: the 2-camera work's handoff, with the full findings, including
  2-camera prep and its limits. Its paths predate this folder.

Rebuilt with `code/make_camera_subset_movies.py` (cut and predict) and
`code/compare_camera_subsets.py` (compare), PIPELINE.md section 2c.

## four_camera_ensembles_roni/

The same four movies at 4 cameras, predicted by different ensembles and scored
on the pipeline's own self-consistency (rigidity of the wing outline, jitter, how
far smoothing moved the points, spread of the fly's body length). Steadiness is
the selector's objective, not accuracy, so this ranks ensembles, not models.

- `pilot_five_ensembles.md`: the 2-camera work's first comparison of five
  ensembles.
- `fair_test_ensembles.md` and `fair_test_member_usage.md`: the planned
  ensemble (A), with 2-camera members added (B, D) or swapped in at the same size
  (C), and which members the selector picked. C was deployed for 4-camera movies.

Rebuilding these needs every member predicted on the movies (`predict_array.sh`)
and the ensemble rebuilt from the members' saved points for each subset
(`find_3D_points_from_ensemble` in `code/prediction_code_lior/predict.py`).

## hull_comparison/

An earlier comparison of this method with the Hull reconstruction
(`code/comparison.py`): psi distributions over a wingbeat.
`manipulated_05_12_22.hdf5`, the Hull data, is not tracked (900 MB).

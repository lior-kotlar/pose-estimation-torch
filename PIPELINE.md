# Batch pipeline: from raw sparse movies to predicted 3D pose

This guide is the start-to-finish recipe for turning a freshly-imported
experiment (raw `*_sparse.mat` movies + an easyWand calibration) into a batch of
predicted, smoothed 3D pose data plus validation plots — unattended, on the
cluster.

There are two layers:

1. **Prep** — give every movie a raw mp4, clean, prescan, mirror-flip, build
   h5s, verify calibration, and write a manifest of good movies. CPU-only.
2. **Predict** — one GPU job-array task per good movie: 2D detection,
   2D→3D triangulation, smoothing, an mp4, and wing-angle plots.

`pipeline.sh` runs both back-to-back. You can also run each layer by hand.

> All commands are run from the repo root
> `/cs/labs/tsevi/lior.kotlar/pose-estimation-torch`. Python is always the venv
> interpreter: `.env/bin/python` (or `source .env/bin/activate` first).

---

## 0. What you need before you start

- An **input directory** of raw movies. Two layouts are auto-detected:
  - **multi-movie**: `<input_dir>/mov1/`, `mov2/`, … each holding the
    `*_cam<N>_sparse.mat` files. (This is the normal case for an experiment.)
  - **single-movie**: `<input_dir>` itself holds the `*_sparse.mat` files and
    its basename is `mov<N>`.

  **Camera count** is detected from the number of `*_sparse.mat` per movie
  dir: **4** for the current rig, **3** for the older one that predates the
  fourth camera. Every movie in an experiment must agree, or prep aborts
  (a movie missing one camera's export would otherwise build a box with a
  blank camera). `--num-cams` overrides. The count flows on its own from
  there — the MATLAB builder, the calibration, verify, triangulation and the
  mp4 all size themselves from the data. Two things to know for a 3-camera
  experiment:
  - the easyWand `.mat` must itself describe 3 cameras;
  - watch `--verify-threshold`. Verify's leave-one-out check triangulates from
    the *remaining* cameras, so on 3 cameras it works from 2 views instead of
    3 and roughly doubles the reported error. Measured on a healthy 3-camera
    experiment that is still only ~6–8 px, well inside the 15 px default — so
    leave it alone unless a run says otherwise.
- The **easyWand calibration `.mat`** for that experiment. For multi-day Roni
  experiments use the END-of-experiment easyWand mat for every movie
  (see the project calibration convention).
- The **mirror camera name** for this rig: `cam1` for 2023 data, `cam5` for
  2022 data (cam5 is mirrored — both the dataset h5 and the calibration must be
  vertically flipped). Older rigs are not on that list, and the mats carry no
  `isFlipped` marker — so **you no longer have to know**: prep runs a MIRROR
  CHECK before the flip that tests every hypothesis against the calibration and
  refuses to proceed if `--cam` disagrees with it. Pass `--cam auto` to let it
  decide outright. It doubles as the idempotency guard the flip has never had:
  the flip is self-inverse, and the check is what stops a second `--cam` run
  from silently un-flipping an experiment.
- A **predict config** JSON — `predict_configurations/config1.json` covers
  every experiment; there is no per-rig variant to choose between. Paths in it
  are repo-relative. `data directory`, `general run name`, `calibration path`
  and `pipeline timings path` are overwritten per-task by the launcher.

  `number of cameras` and `calibration path` should normally be left at
  `"auto"`: the count is read from the movie's own box (`cropzone`) and the
  calibration is taken from the experiment directory holding the movie, so
  one config covers both the 3-camera and 4-camera rigs and cannot silently
  triangulate against a stale calibration. The ensemble members follow from
  that count — each `prediction_models/*/model.json` declares `num cameras`
  (or nothing, meaning any), and members that cannot run on the movie are
  reported and skipped. An explicit value still wins, but one that
  contradicts the box is a hard error rather than a wrong answer.

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

1. The CPU prep job runs `process_experiment.py` (raw movies → clean →
   prescan → mirror check → flip → build → verify → manifest), appending
   per-step timings to `<input_dir>/pipeline_timings.csv`.
2. If `manifests/good_movies_<experiment_name>.txt` ends up non-empty, the job
   submits `predict_array.sh` as a **separate** GPU array job, sized to the
   manifest and named after `-J` so every task lands under one run directory.
3. The prep job exits as soon as the array is submitted; the array runs in
   parallel on GPU nodes.

`-J <experiment_name>` becomes both the prep log name and the predict run name
(the parent output directory). The optional 5th arg caps concurrent GPU tasks
(default 32).

Two environment variables tune the predict array without editing any script
(set them in the shell before `sbatch`; SLURM carries them into the job, and
`pipeline.sh` carries them on to the array it submits):

| variable | effect |
|---|---|
| `PREDICT_SBATCH_ARGS` | extra `sbatch` options for the predict array, e.g. `"-p catfish,salmon --gres=gpu:1 --mem=96g --cpus-per-task=12"`. `predict_array.sh`'s own defaults (256 GB, 32 CPUs, an L40S on salmon) are sized for full 5500-frame movies and can queue behind a deep salmon backlog; measured peak RSS is ~8.4 MB per frame + ~3 GB, so 96 GB covers any 5527-frame movie. |
| `DROP_BOX_CACHE=1` | each predict task deletes its movie's `saved_box_dir` after a **successful** prediction. It is a regenerable cache and ~236 kB/frame — over half of the ~413 kB/frame a predicted movie costs in total (built h5 ~85, cache ~236, `predict_output` ~92). |

Prep itself needs little memory (peak ~1.8 GB) but can run long: builds are
serial across experiments (concurrent prep jobs stall each other on MATLAB), so
chain experiments with `--dependency=afterany:<previous job>` and raise
`--time` for large ones — `pipeline.sh` defaults to 12 h. The first prep of an
experiment also builds every movie's raw movie, ~6 min a movie; later runs find
them and skip the step.

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
| `--num-cams N` | override the detected camera count (3 or 4) |
| `--prescan-min-intersection N` | min all-4-cam single-fly run to keep a movie (default 500) |
| `--prescan-min-edge-margin N` | px of clearance the fly must keep from every image border (default 5; 0 disables) |
| `--prescan-min-cams-in-frame N` | how many cams must see the WHOLE fly for a frame to count (default 3; 0 = every cam) |
| `--prescan-only` | only run the prescan, then stop |
| `--verify-only` / `--no-verify` | run only / skip the reprojection sanity check |
| `--verify-threshold PX` | flag movies whose reprojection error exceeds PX (default 15) |
| `--cam auto` | let the mirror check pick the camera to flip |
| `--no-mirror-check` | skip the pre-flip verification (removes the guard) |
| `--skip-clean / --skip-flip / --skip-build` | skip individual stages |
| `--skip-raw-movies` | don't build the raw movies that are missing (built by default, before every other stage) |
| `--perturbation` | declare this a perturbation experiment (see below) |
| `--perturbation-type` | e.g. `roll`, `yaw` |
| `--perturbation-onset-frame N` | trigger-relative onset (default 0) |
| `--perturbation-duration-ms X` | omit when the log never recorded it |
| `--dry-run` | print what would happen, change nothing |

Outputs of prep:
- one `<movie>_raw_fr30_skip1.mp4` per movie dir, beside its mats (the raw
  movie, built only where there was none; MATLAB's output in `raw_movie.log`),
- one `mov_<n>_<start>_<end>_ds_*tc_*tj.h5` per movie (the dataset h5),
- one `<movie_dir>/prescan_cam_validity.npz` per movie (which cams saw the
  whole fly at each built frame),
- one shared `<input_dir>/calibration.h5`,
- `manifests/good_movies_<experiment>.txt` (the good-movie list),
- `<input_dir>/process_report.txt` (prescan + verify transcript),
- `<input_dir>/pipeline_timings.csv` (per-step timings).

Inspect `process_report.txt` and the prescan/verify output before predicting.

The prescan's `out-of-frame` line is worth reading: it counts, per camera, the
frames where the fly's blob ran into an image border. Those frames hold a
truncated fly and the network's 2D detections on them are meaningless.

A frame does **not** need every camera to see the fly whole — only
`--prescan-min-cams-in-frame` of them (default 3). The count is what matters,
so the majority may be a different set of cameras each frame. The
`whole-fly cams per frame` line reports the distribution, and when the rule is
relaxed the per-movie line also states what the strict all-cams rule would
have given, so the trade is visible without a second scan.

That relaxation is only safe because the prescan also records *which* cameras
were whole, in `<movie_dir>/prescan_cam_validity.npz`. Prediction reads it and
drops every camera *pair* containing a cut camera for that frame, so a
truncated fly can no longer poison the 3D pose. Movies built before this
existed can be back-filled without rebuilding:

```bash
.env/bin/python code/rebuild_edge_cut_movies.py <manifest> --mask-only
```

A frame where too few cameras see the fly whole still ends the build range, so
a high out-of-frame count means the movie got trimmed — not that anything is
wrong.

### 2a-bis. Perturbation experiments

Add `--perturbation` to the prep step and it writes a `perturbation.json` next
to `calibration.h5`. That file's **presence is the label** — predict picks it
up on its own, so nothing has to be repeated in the predict config:

```bash
sbatch -J <name> sbatch_files/pipeline.sh <input_dir> <easywand> none \
    predict_configurations/config1.json 24 \
    --perturbation --perturbation-type roll --perturbation-duration-ms 7.5
```

(anything after the 5th argument is forwarded verbatim to
`process_experiment.py`.) An **existing** `perturbation.json` is validated and
kept, not overwritten — a hand-authored one can carry per-movie windows the
CLI cannot express. `--perturbation-force` replaces it.

Every predicted movie then gets, in its `*_analysis_smoothed.h5`:

| dataset | meaning |
|---|---|
| `perturbation_declared` | 1 — a declaration applied to this movie. **Its absence is how "nothing was declared" is expressed** |
| `perturbation_status` | `perturbed` / `control` / `unknown` |
| `perturbation` | 1 only when there is a window to draw; 0 for control and unknown |
| `perturbation_type`, `perturbation_type_known` | e.g. `roll`; `_known` is 0 when the log never named a type |
| `perturbation_start_frame` | onset, trigger-relative — always known when perturbed |
| `perturbation_start_index` | row holding the onset, or `-1` if outside this movie |
| `perturbation_end_known` | 0 when no duration is known at all |
| `perturbation_end_frame` / `_end_index` | only when the end could be located in frames |
| `perturbation_duration_ms` | whenever a duration is known — recorded **or** assumed |
| `perturbation_duration_source` | `recorded` / `assumed` / `unrecorded` / `n/a` |
| `perturbation_duration_assumed_ms` | only when the duration was assumed |
| `perturbation_duration_note` | the declaration's own provenance sentence |
| `perturbation_frames_trigger_relative` | 0 when the trigger could not be established |
| `perturbation_source`, `perturbation_movie_key` | which file and which per-movie entry applied |
| `perturbation_state` | per frame: 0 before, 1 during, 2 after, 3 control, **-1 unknown** |

**The four states a reader must be able to tell apart**

| on disk | means |
|---|---|
| no `perturbation_declared` | nothing was declared for this movie |
| `declared=1, perturbation=1, status=perturbed` | it was perturbed |
| `declared=1, perturbation=0, status=control` | declared **unperturbed** — an experimental control |
| `declared=1, perturbation=0, status=unknown` | declared, but the status itself is not known |

**Mixed experiments.** One experiment can hold both perturbed and unperturbed
movies — 030121 is exactly that, a chamber of flies some of which carried a
magnet. A `movies` block keyed by movie-directory basename overrides the
experiment-level block key by key:

```jsonc
{
  "schema_version": 2,
  "perturbation": {"type": "yaw", "status": "perturbed",
                   "onset_trigger_frame": 0, "duration_ms": 7.5,
                   "duration_source": "assumed"},
  "movies": {
    "mov9":  {"status": "perturbed"},
    "mov16": {"status": "control", "evidence": "no magnet on this fly"},
    "mov2":  {"status": "unknown", "evidence": "absent from the log"}
  }
}
```

A file with no `status` anywhere resolves exactly as it did before, so old
declarations keep working unchanged.

**Assumed durations.** When a movie is known to be perturbed but the log never
recorded how long the pulse lasted, `utils.PERT_DEFAULT_DURATION_MS` (7.5 ms,
the rig's pre-set per the thesis) is applied and marked
`duration_source: "assumed"`. It is never silent: the word travels into the h5,
the CSV, both PNG subtitles, the mp4 counter and the viewer header, the
shaded band is drawn hatched, and `"assume_duration": false` opts out and
restores the honest `unrecorded`.

plus, in the CSV, a per-row `perturbation_state`
(`before`/`during`/`after`/`control`/`unknown`), the constant columns
`perturbation_status`, `perturbation_type`, `perturbation_onset_frame`,
`perturbation_end_frame`, `perturbation_duration_ms` and
`perturbation_duration_source`, and the per-row `frames_from_onset` /
`time_from_onset_ms`.

In the mp4 the counter gains a **static** line naming the perturbation
(`yaw | onset trigger frame 960 | end 1080 (7.50 ms, ASSUMED)`) above the
per-frame `PRE -7.50 ms` / `PERT +0.25 ms` / `POST +0.75 ms` line — the type
appears in the video for the first time. A control movie shows
`CONTROL (no perturbation)`; an unknown one `PERTURBATION STATUS UNKNOWN`.
Both PNGs carry the same line as a subtitle and the viewer as its header.

`DURING` is the half-open interval `[onset, end)`: the frame at `end_frame` is
already `after`, in every product.

**`-1` / `unknown` is a real answer, not a gap.** When the onset was logged but
the duration never was, frames before the onset are still labelled exactly;
frames from the onset on are genuinely indeterminate, and calling them "not
after" would assert more than the record supports.

The window is read at predict time, so **changing `perturbation.json` after a
run means re-predicting those movies.** Get the declaration right before
launching; if the duration is genuinely unrecorded, `unknown` is the correct
thing to ship rather than a placeholder to fix up later.

Prep also prints a **PERTURBATION COVERAGE** section. The prescan picks its
build range from fly visibility and knows nothing about the perturbation, so
some movies get clipped to start after the onset and hold no pre-perturbation
baseline. That count is worth reading before the GPU array runs.

### 2a-ter. Lighting — a second stimulus axis

The same `perturbation.json` carries a `lighting` block (schema 3), resolved per
movie exactly like the pulse (a `movies[<dir>].lighting` entry overrides it):

```jsonc
"lighting": {
  "regime": "darkening",            // constant_light | constant_dark | darkening | unknown
  "light_off_trigger_frame": 0,     // darkening only: the frame the white light goes OFF
  "relight_after_ms": 1000,         // darkening only (default: the rig's 1 s)
  "note": "...", "evidence": "..."  // provenance, in words
}
```

`darkening` means the light is switched off *during the recording* — a visual
perturbation in its own right, distinct from `constant_dark` (dark all session,
no light change inside any movie). A file with no `lighting` block reads as
**not declared**, never as lit. At prep, `--lighting-regime`, `--light-off-frame`,
`--relight-after-ms` and `--lighting-note` write the block (CLI declarations only;
an existing file is kept as usual).

Every product then states it:

| product | what it shows |
|---|---|
| `*_analysis_smoothed.h5` | `lighting_declared`, `lighting_regime`, `lighting_darkening` (strict 0/1), `lighting_light_off_frame`/`_index`, `lighting_light_on_frame`/`_index`, `lighting_relight_after_ms`, `lighting_note`, `lighting_evidence`, `lighting_frames_trigger_relative`, per-frame `lighting_state` (0 lit, 1 dark, -1 unknown) |
| CSV | `lighting_regime`, `light_off_frame`, per-row `light_state` (`lit`/`dark`/`unknown`), `frames_from_light_off`, `time_from_light_off_ms` |
| `wing_angles.png`, `body_angular_acceleration.png` | a strip along the top of every panel (amber LIGHT ON / black DARK / grey LIGHT ?), a dash-dot light-off line and grey wash for a darkening, and a lighting line in the subtitle |
| `*_flight_viewer.html` | the same strip and line on every time-series row, a lighting badge in the header, and a readout line (`DARK +12.50 ms since light-off`) whose box turns dark on dark frames |
| `movie 2D and 3D.mp4` | a per-frame lighting line under the pulse line; the counter box turns dark on every dark frame |
| `All body data.html`, `movie_html.html` | the pulse and lighting as title lines, the strip/band/line as shapes (All body data), and light-off / pulse onset / pulse end markers on the 3D trajectory (movie_html) |
| `source.json` | the resolved `perturbation` and `lighting` blocks |

**Which frame is 0.** Frame 0 is the **camera trigger** in every product --
the h5 `frame_index`, the CSV `frame` column, the mp4 counter, both PNGs and the
flight viewer. The rig's Arduino schedules both stimuli from that same trigger,
so the pulse sits at frame 0 only in experiments that fire it on the trigger
(ex210824/26 fire it at 960, 60 ms later, and switch the light off at 0). The
figures draw the pulse and the light-off as events at their own frames, and
their texts add the timing in words ("yaw pulse, 7.50 ms long (ASSUMED): frames
960-1079, starting 60.00 ms after the camera trigger"; "light switched OFF at
frame 0 (the camera trigger), 60.00 ms before the pulse"). The viewer's "from"
menu can still redraw the axis from the pulse onset, and `plot_wing_and_body.py
--origin perturbation` does the same for the PNGs; the axis label then names its
zero. The CSV's `frames_from_onset` / `frames_from_light_off` give the other
two references per row.

DARK is the half-open interval `[light_off, light_on)`, the same convention as
the pulse's DURING. Constant regimes are labelled even without a trigger; a
darkening without trigger-relative frames is `unknown`.

**Changing the declaration after prediction does not need the GPU.**
`reanalyse_movies.py` (and its array wrapper `sbatch_files/reanalyse_array.sh`)
rewrites every derived product — h5, CSV, PNGs, viewer, both plotly pages,
source.json and, with `--with-mp4 --force-mp4`, the mp4 — from the cached 3D
points, reading the **live** `perturbation.json` (`--perturbation-source auto`):

```bash
# manifest = one predicted movie OUTPUT dir per line
N=$(wc -l < manifests/reanalyse_X.txt)
sbatch -J reanalyse_X --array=0-$((N-1))%40 sbatch_files/reanalyse_array.sh manifests/reanalyse_X.txt
```

A whole experiment also runs as one job on one node: `--jobs N` re-analyses N movies at once,
`--only-stale` skips movies whose products the current code and declaration already made (so a
re-run resumes), and `--dry-run` is a preflight that writes nothing. Movies kept on a PC are
re-analysed there, collected and uploaded to `collected_h5` with no cluster job at all; see
[LOCAL_REANALYSIS.md](LOCAL_REANALYSIS.md). Their ensembles can be re-run from there too, on the
cluster, with `local_reanalysis/realign.bat` (below).

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
    <mov>_analysis_smoothed.csv                    # per-frame body+wing state
    <mov>_analysis_smoothed_flight_viewer.html     # interactive 3D + graphs
    All body data.html / movie_html.html           # plotly summaries
    source.json                                    # provenance
    wing_angles.png, body_angular_acceleration.png
```

`<run_name>` is the `-J` job name (so all movies of one experiment share a
parent). Per-movie/per-step timings accumulate in the experiment's
`pipeline_timings.csv` (`predict`, `plot`, `viewer`, `total` rows joined on
`mov<N>`).

Body pitch -- `pitch_angle` and `pitch_dot` in the h5, `body_pitch_deg` in the
CSV -- is **nose-down positive**: the right-hand rule about `y_body`, which
points left. It shares its sign with `omega_body[:, 1]` (`q`) and
`omega_body_dot[:, 1]`, and a fly holding its nose above the horizon reads a
negative pitch. Analysis files written before this carry no `pitch_convention`
dataset and hold the opposite sign. The plotting tools here flip those on read;
the CSV and any other direct reader do not, so re-run `code/reanalyse_movies.py`
on them.

Body roll -- `roll_angle` and `roll_dot` in the h5, `body_roll_deg` in the CSV --
is measured only once per wingbeat (about every 73 frames): when both wings are
spread sideways, the plane of the wing tips gives the fly's left-right axis
`y_body`. Every other frame comes from a smoothing spline through those
measurements (`join_y_body_measurements`), and frames before the first or after
the last measurement are NaN. So the roll rate `p` (`omega_body[:, 0]`) and
above all the roll acceleration `omega_body_dot[:, 0]` are only partly
measured: about half of the roll acceleration's size depends on how the
measurements are joined, and nothing shorter than a wingbeat or two is
resolved. Don't read a peak roll acceleration during a perturbation pulse as a
measured value.

---

## 4. Quick sanity checks & standalone tools

These power the pipeline but are runnable on their own:

```bash
# Is a movie worth processing? (longest all-4-cam single-fly run)
.env/bin/python code/scan_sparse_movies.py <movie_dir>

# Reprojection-error check of a built h5 against calibration.h5
.env/bin/python code/verify_calibration.py <movie.h5> <calibration.h5>

# Which cam is the mirror? Tests every flip hypothesis against the DLT.
# Runs on the RAW mats -- no MATLAB, no build, nothing flipped. Prep runs this
# automatically now; use it standalone to see the full ranked table, or to
# check an experiment before committing to a prep run.
.env/bin/python code/find_mirror_cam.py <experiment_dir> --easywand <easywand.mat>

# Flip the mirror cam's sparse mat in place (single or batch)
.env/bin/python code/flip_sparse_cam_mat.py <movies_dir> --cam cam1 --dry-run

# Raw movies: every mov<N>/ under a folder (sub-folders included) without a
# <movie>_raw_fr30_skip1.mp4 gets one -- the step prep runs first. A big tree
# goes on a CPU job array: each task takes its own share, so tasks never overlap, and
# re-submitting only fills in what is still missing (~6 min a movie).
.env/bin/python code/make_raw_movies.py <folder> --dry-run
sbatch -J raw_<name> --array=0-19 --gres=gpu:0 --mem=16g --mail-type=FAIL \
    sbatch_files/sbatch_configurable.sh code/make_raw_movies.py <folder>

# Wing-angle + body angular acceleration plots from an analysis h5 (or a dir of them).
# The x-axis is trigger-relative, or zeroed on the perturbation onset when the
# movie declares one (--origin trigger keeps the mp4 counter's numbering).
.env/bin/python code/plot_wing_and_body.py <dir>

# 3D check of the gravity ("down") vector: body triad + gravity every k frames
.env/bin/python code/plot_gravity_body.py <dir> -k 100

# Re-run the ensemble step for movies predicted before the pose models' wing labels were
# aligned (wing_labels.harmonize_wing_labels). CPU only, from the saved per-model candidates;
# installs the new 3D points only where nothing got worse, then re-analyses. --dry-run lists
# which movies would change at all. Any further argument is passed on to realign_ensemble.py.
.env/bin/python code/realign_ensemble.py --list <manifest> --dry-run
sbatch --array=1-$(wc -l < <manifest>) sbatch_files/realign_ensemble_array.sh <manifest>
# Movies kept on a PC take the same route without anyone touching the cluster by hand: their
# owner runs local_reanalysis/realign.bat, which uploads only the ensemble members of the
# movies that would change, submits this same array job, brings the new points back and
# re-analyses them there (LOCAL_REANALYSIS.md).

# Interactive viewer: the fly flying through the lab frame, scrubbable, beside
# two panels of analysis signals (--rows for more) -- time series, or one wing's
# path through angle space in 3D or in any of its three 2D projections. One
# self-contained ~8 MB HTML per movie (--cdn halves it but then needs a network
# connection to open). Written automatically by predict; run it standalone to
# rebuild one, or a whole run at once.
.env/bin/python code/plot_flight_viewer.py <dir>

# Shrink an h5 to its first N frames (fast iteration)
.env/bin/python code/truncate_h5_movie.py <movie.h5> 1500
```

MATLAB build/calibration scripts (driven by `code/build_experiment.sh`, but
overridable from the CLI) live in `matlab/` and addpath into the vendored
`micro-flight-lab-master/` for `HullReconstruction`. The raw-movie renderer is
`matlab/+VideoEditing/JoinSparses.m`.

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

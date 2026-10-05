# Re-analysing predicted movies on your own PC

Use this when movies were **already predicted** (they have 3D points), the prediction folders
are on a disk at your PC, and you want their flight data rebuilt with the current code, without
uploading the movies or predicting again.

You do the setup once. After that, everything is one thing: **drag a folder onto
`reanalyse.bat`**. It looks at every movie in that folder, works out which of three jobs each one
still needs, and does only those:

| | what it fixes | where it runs |
|---|---|---|
| **realign** | one of the pose models labelled the two wings the other way round, so the ensemble mixed them into one physical wing | the lab cluster |
| **re-analyse** | the analysis h5, CSV, plots, flight viewer and plotly pages are older than the code, the declaration or the 3D points | your PC |
| **render** | the overlay mp4 was made from points the movie no longer has | the lab cluster |

A movie that needs nothing is skipped in no time. An experiment that only needs its videos redone
never waits for anything else. And the three are connected the way you would expect — repairing an
ensemble changes the 3D points, which makes the analysis out of date, which makes the video out of
date — so asking for one can pull in the others, in that order.

Before doing anything it prints what it found, so you always see the size of the job first.

**Movies not predicted yet?** If you have an experiment's raw movies (the `*_sparse.mat` files) on
this PC, `predict.bat` preps and predicts them on the cluster and brings the results home, with the
same setup. See [LOCAL_PREDICT.md](LOCAL_PREDICT.md).

---

## What you need

- A Windows 10 or 11 PC.
- The predicted movie folders on a disk. The folder you give can be one experiment, or a folder
  holding many experiments, at any depth.
- An account on the lab server (`moriah-gw-01.cs.huji.ac.il`).
- **Only for rebuilding videos:** the source datasets those movies were built from — the folder
  holding each experiment's box h5 files and its `calibration.h5`. Setup asks where it is. Without
  it everything else still works, and the movies whose video cannot be rebuilt are named.

---

## One-time setup (about 15 minutes)

### 1. Install Python 3.11

Download **Windows installer (64-bit)** from
https://www.python.org/downloads/release/python-3119/ and run it. Keep the default options and
click **Install Now**.

It has to be 3.11. Other Python versions you already have don't matter, since the tool always
picks 3.11.

### 2. Download the tool

Open **Command Prompt** (Start menu → type `cmd`). Paste these four lines, replacing
`YOUR_USERNAME` with your server username:

```bat
mkdir C:\pose-reanalysis
ssh YOUR_USERNAME@moriah-gw-01.cs.huji.ac.il "SLURM_CONF=/vol/slurm/moriah/slurm.conf /vol/slurm/moriah/bindir/bin/srun --ntasks=1 --mem=2g --time=0:10:00 --gres=gpu:0 --chdir=/tmp --job-name=pose_setup git -C /cs/labs/tsevi/lior.kotlar/pose-estimation-torch -c safe.directory='*' archive --format=tar HEAD code local_reanalysis requirements-analysis.txt LOCAL_REANALYSIS.md LOCAL_PREDICT.md" > C:\pose-reanalysis\download.tar
tar -xf C:\pose-reanalysis\download.tar -C C:\pose-reanalysis
del C:\pose-reanalysis\download.tar
```

The first time you connect, ssh asks `Are you sure you want to continue connecting`: type
`yes`. Then type your server password. Nothing shows while you type; that's normal.

`srun` in there is deliberate: `moriah-gw-01` is only the way in, and nothing may run on it, so
every command is handed to the cluster's scheduler and runs on a compute node. The scheduler is
named in full, with its configuration file, because the shell you land in on the gateway has
neither on hand. You may see
`srun: job 123456 queued and waiting for resources` for a moment; that is the wait for a free
node, and the download carries on by itself.

Afterwards `C:\pose-reanalysis` contains `code`, `local_reanalysis`, `LOCAL_REANALYSIS.md`,
`LOCAL_PREDICT.md` and `requirements-analysis.txt`. In `local_reanalysis` you will find the files
you double-click or drag folders onto: `setup.bat`, `reanalyse.bat` (the everyday one), `check.bat`,
`realign.bat`, `render.bat`, `reanalyse_including_bad.bat` and `update.bat`, plus `predict.bat` and
`predict_check.bat` for raw movies ([LOCAL_PREDICT.md](LOCAL_PREDICT.md)).

### 3. Run the setup

Double-click **`C:\pose-reanalysis\local_reanalysis\setup.bat`**. It:

1. creates a private Python environment in `C:\pose-reanalysis\venv`,
2. installs the packages it needs (about 1 GB, a few minutes),
3. asks for your **server username** (Enter keeps the suggested server address),
4. asks where you keep your **source datasets** — the folder holding each experiment's movies as
   they came off the rig. Only rebuilding a video needs it; leave it empty if you do not have them
   on this PC, and everything else still works,
5. asks whether to log in **without a password** from now on. Answer `y`, then type your server
   password one last time,
6. checks that it can reach the server, and whether this account may publish results to it.

It ends with `Setup finished.` Press any key to close the window.

**Only the pipeline's owner uploads.** If your account cannot write the upload folder on the
server, setup says so and turns uploading off for this PC:

```
This account cannot write to /cs/labs/tsevi/lior.kotlar/pose-estimation-torch/collected_h5,
so this PC will re-analyse and collect movies for itself only -- nothing is uploaded.
```

Such a PC still re-analyses its movies in place and collects them into
`C:\pose-reanalysis\collected_h5`; what it cannot do is the two stages that run on the cluster.
`check.bat` still tells you which movies need them — send that list on. To hand results over, give
the owner the collected folder (or the movie folders themselves).

---

## Running it

**Drag the folder** onto `C:\pose-reanalysis\local_reanalysis\reanalyse.bat`.
Or double-click `reanalyse.bat` and paste the folder's path when asked.

Tip: right-click `reanalyse.bat` → *Send to* → *Desktop (create shortcut)*. You can then drop
folders on the desktop icon.

It always starts by working out what each movie needs, and shows you before it does anything:

```
=== working out what each movie needs ===
248 movie(s) in 9 experiment(s)
9 declaration file(s) found on the server
analysis code 5c8540b2fff20855 (commit c870bb1)

experiment                       movies  realign  re-analyse  render  nothing
Tsory/ex201224_dark_roll_t0_7ms       8        0           0       8        0
Tsory/ex210825_dark_yaw_t0          206       31         206     206        0
roni_dark/2023_08_07_10ms             3        0           0       0        3

to do   : realign 31, reanalyse 206, render 217
skipping: 25 movie(s) already up to date
```

Then it does the stages that have work in them, in that order, asking again after each one — so
nothing is done twice and nothing is done for a movie that turned out not to need it.

| stage | how long |
|---|---|
| realigning, on the cluster | about half an hour a movie, many at once |
| re-analysing, here | 10–20 seconds a movie |
| rendering, on the cluster | about 20 minutes per thousand frames, many at once |
| collecting and uploading | seconds |

**To see the list without doing anything, use `check.bat`.** It is safe to run at any time, on any
PC: it reads your own disk, changes nothing, and uploads nothing.

**To do just one stage**, use `realign.bat` or `render.bat`. The everyday command is
`reanalyse.bat`, which does whichever of them a movie needs.

**Closing the window is always safe.** A cluster stage is written down as it goes, so running the
same command again picks the round up where it stopped rather than starting it over. Movies
already done are skipped in no time.

### Movies in `bad_signal` and `bad_wings` folders

They **are** re-analysed like every other movie: their h5, plots, viewer and pages are rebuilt
in place. What they are left out of is the collecting and uploading, so a known-bad movie never
reaches the server among an experiment's usable files. The run says how many were left out.

To include them, drag the folder onto **`reanalyse_including_bad.bat`** instead (or add
`--include-bad`). They are then collected and uploaded under their experiment's own
`bad_signal\` or `bad_wings\` subfolder — for example
`collected_h5\Tsory\ex210825_dark_yaw_t0\bad_signal\` — so copying an experiment's files
onward still never picks them up by accident.

**Running it again is always safe.** Movies already made by the current code are skipped in
0 seconds, and only files the server doesn't have yet are uploaded. So if the PC sleeps, the
window is closed, or the connection drops, **just run the same thing again** and it continues
where it stopped.

### The report

`C:\pose-reanalysis\reports\reanalyse_report_<time>.csv` has one row per movie:

- **`status`:** `done` (redone now), `current` (already up to date) or `failed`.
- **`max_abs_change_yaw_deg`, `max_abs_change_pitch_deg`, `max_abs_change_roll_deg`:** how far
  the body angles moved compared with the previous version.
- **`wing_nan_frac_old`, `wing_nan_frac_new`:** the share of frames with no valid wing angle,
  before and after.
- **`check`:** filled in when something moved that shouldn't have (yaw or pitch), or more wing
  frames became invalid. Look at those movies' plots against the `superseded_…` versions.

Next to it, `run_<time>.log` holds everything the run printed, including each movie's messages.
If anything looked wrong, that file says what happened, even after the window is closed.

---

## The two jobs that run on the cluster

Both need more than your PC has, so `reanalyse.bat` sends what they need, has the cluster do the
work, and brings the result back into the movie's own folder. Both are written down as they go, so
closing the window costs nothing.

### Repairing wings labelled the wrong way round

Every movie is predicted by several models at once, and their answers are combined into one set of
3D points — the **ensemble**. In some movies one model labelled the fly's two wings the other way
round from the rest. The combining step then averaged that model's "left wing" with the others'
left wing, and the result put **both wings, and both hinges, on one physical wing** for part of the
movie. The wing angles of such a movie are unusable.

Predictions made after 14 September 2026 already have this fixed. Older ones do not, and
**re-analysing cannot repair them**: the damage sits in the 3D points, which the analysis only
reads. The only cure is to combine the models again, with the labels aligned first — about half an
hour of computing a movie, which is why it runs on the cluster and only for the movies that need
it (about one in six of the ones checked so far).

Only the ensemble members travel, about 18 MB a movie. Nothing else is needed: not the source
movie, not the calibration, not the video.

#### A movie is changed only when nothing got worse

Re-combining is not automatically better, so every movie is judged before anything is replaced.
The cluster compares the old and the new points on four things: how many frames have both wings on
one wing, how many wing stroke angles come out impossible, how much each wing's shape wobbles from
frame to frame, and the body's pitch and yaw, which should not move at all.

If any of those got worse, **the movie is left exactly as it was** and the run says why:

```
  Tsory\ex210103_dark_yaw_t0_mixed\mov_21: LEFT ALONE, BLOCKED: more out-of-range phi
```

The refusal is remembered in the movie's folder (`.realign_blocked.json`, which also holds the
numbers behind it), so later runs leave it out instead of spending another half hour being refused
again. `--retry-blocked` offers those movies once more. Such a movie is not re-analysed or
re-rendered either, because nothing about it changed.

### Rebuilding the overlay video

`movie 2D and 3D.mp4` is drawn from the 3D points reprojected onto the camera images. When the
points change — a repair, or an analysis that decides left from right differently — the video no
longer matches the h5 beside it, and after a repair it can be showing both wings collapsed onto one
while the data says otherwise.

Rendering needs the **camera images**, which are in the box h5 your movies were built from. That is
the one thing a movie folder does not contain, so this is the only stage that needs your datasets
folder (setup asks for it once). The renderer reads one time-channel per camera out of nine, so
only those are sent: about 60 MB a movie instead of 185 MB. The finished mp4 comes back and the old
one is kept beside it.

A movie predicted with `predict.bat` came home with exactly those channels already, as
`<movie>_render.h5` beside its mats; it is found and sent as it is.

A movie whose box h5 cannot be found is reported and skipped; nothing about it is changed:

```
cannot render 9 movie(s): their source box h5 was not found (--dataset-root says where to look)
```

**Videos made before this existed cannot be judged.** Nothing in an older movie folder records what
its video was made from, so those are listed as unsure and left alone rather than redone blindly —
ten hours of cluster time for videos that are mostly fine. `render.bat <folder> --render-unknown`
redoes them anyway. From now on each video carries a `video.json` saying exactly which points it
came from, so the question answers itself.

### What changes in a movie's folder

| file | when |
|---|---|
| `points_3D_smoothed_ensemble_best_method.npy` and the model-selection files | a repair |
| `<movie>_analysis_smoothed.h5`, `.csv`, `wing_angles.png`, `body_angular_acceleration.png`, the flight viewer, the plotly pages, `source.json` | a re-analysis |
| `movie 2D and 3D.mp4`, `points_ensemble_smoothed_reprojected.npy`, `video.json` | a render |
| `superseded_<date>_<time>\` and `superseded_ensemble_<date>_<time>\` | **the previous version of everything above**; nothing is ever deleted |

The predictions of each individual model are never touched — only the combination of them.

---

## Where the collected h5 files go

On the PC (`C:\pose-reanalysis\collected_h5`) and on the server
(`/cs/labs/tsevi/lior.kotlar/pose-estimation-torch/collected_h5`), the files are sorted the same
way: one folder per experiment, named after the experiment's folder under the lab's
`inference_datasets`:

```
collected_h5\
  Tsory\
    ex201224_dark_roll_t0_7ms\
      mov_1_496_1439_ds_3tc_7tj_analysis_smoothed.h5
      ...
      superseded_20260920_101500\     older versions, when a movie was re-analysed again
    ex210825_dark_yaw_t0\
  roni_dark\
    2023_08_06_40ms\
  local_only\
    my_experiment\
```

- **The folder comes from each movie's own records, not from how you named your folders**, so
  the same movie always lands in the same place. Build batches such as `1to30` are merged into
  their experiment.
- **`local_only\<name>`** holds movies that record no experiment at all. `<name>` is the folder
  the movie folder sits in.
- **Movies in `bad_signal` or `bad_wings` folders** are re-analysed but not collected, unless
  you ask for them (see above); then they sit in a `bad_signal\` / `bad_wings\` subfolder of
  their experiment.
- **Movies that failed** are not collected, so an old h5 is never uploaded as if it were new.
- **When an h5 on the server is replaced**, the old one moves into `superseded_<time>\` next to
  it.

---

## Keeping the tool up to date

**This happens by itself.** Every run first asks the server whether the code has changed. If it
has, the run says so, updates itself (installing new packages if the list changed, keeping your
settings), and then starts with the new code:

```
the server has newer code (a1b2c3d -> e4f5g6h); updating before the run
code updated to commit e4f5g6h

starting the run with the updated code
```

So you never have to pull anything by hand. Two things follow from it:

- When the **analysis** code changes, the next run re-analyses every movie, because what the
  products contain is decided by the code that made them. That is the point of the check: it stops
  you from re-analysing with an old copy and having to do it again later. A change that does not
  touch the analysis costs nothing — the survey still reports those movies as up to date.
- Videos are **not** redone by a code change on its own. A video is only out of date when the
  points it was drawn from have moved, which is a question the survey answers from the movie's own
  files.
- `local_reanalysis\update.bat` still exists if you want to update without running anything, and
  `reanalyse.bat <folder> --no-update` runs with the copy you have.

---

## If something goes wrong

| you see | what to do |
|---|---|
| `'py' is not recognized` or `Python 3.11 was not found` | Install Python 3.11 (setup step 1), then run `setup.bat` again. |
| `'ssh' is not recognized` or `the 'ssh' command was not found` | Windows Settings → System → Optional features → add **OpenSSH Client**. |
| It asks for the server password every time | Run `setup.bat` again and answer `y` to logging in without a password. |
| `STOPPED: could not download the declarations` | The server couldn't be reached. Nothing was changed. Check your internet or VPN and run again. |
| `STOPPED: the upload was cut off` / `did not accept the upload` | The analysis and the local collection are kept. Run again; only what's missing is sent. |
| A figure or page is missing for a movie | Its message is in `C:\pose-reanalysis\reports\run_<time>.log`; one figure failing never stops the rest. |
| A movie `FAILED` with **"does not record where the camera trigger is"** | Its previous analysis is too old to place frame 0 at the camera trigger, so it is skipped rather than numbered wrongly. Ask Lior. |
| Any other `FAILED` movie | The message above it and the `error` column of the report say why. The other movies are unaffected. |
| `no predicted movies ... under` | That folder has no movie folders with `points_3D_smoothed_ensemble_best_method.npy` in them. Check the path. |
| It says **only the pipeline's owner** can do a stage | Expected on any PC but Lior's: realigning and rendering both compute on the cluster. The list of affected movies it printed is the useful part; send it on. Re-analysing and collecting still run. |
| A movie says `LEFT ALONE, BLOCKED: ...` | Re-combining that movie would have made something worse, so it was not touched, and it is not re-analysed or re-rendered either. Nothing to undo. |
| A cluster stage stops midway (window closed, connection lost) | Run the same command on the same folder again: it picks the round up where it stopped and never repeats work already done. |
| `the cluster would not start this round` | The cluster refused the job (usually a full queue or a full disk). Nothing on the PC changed; try again later. |
| A step sits at `queued and waiting for resources` | Normal: the command is waiting for a free compute node, because nothing may run on the gateway. It continues by itself. |
| `the lab filesystem is not mounted on <node>` | That node came up without `/cs/labs/tsevi`. The tool already waited and tried again; run the same command once more. |
| `cannot render N movie(s): their source box h5 was not found` | Point `dataset_root` at the folder holding your experiments' source data (run `setup.bat` again, or edit the settings file). Everything else still runs. |
| `unsure about the video of N movie(s)` | Those videos predate the stamp that says what a video was made from, so nothing on disk can judge them. They are left alone; `render.bat <folder> --render-unknown` redoes them. |
| A stage you expected does not run | `check.bat` prints why: a movie only appears under a stage when its fingerprints say it is out of date. |
| `connected, but the check did not come back` | Either the project path is wrong, or the scheduler cannot be reached from where you log in. Check the `server_project` setting, and that this answers: `ssh <server> "SLURM_CONF=/vol/slurm/moriah/slurm.conf /vol/slurm/moriah/bindir/bin/srun --version"`. |
| `srun: fatal: Could not establish a configuration source` | The shell you landed in has no `SLURM_CONF`. The tool sets it itself; if you are typing a command by hand, put `SLURM_CONF=/vol/slurm/moriah/slurm.conf` in front of `srun`. |

To stop a run, close the window or press Ctrl+C. Run it again later to continue.

---

## Details, for the curious

**What it reads per movie.** Only `points_3D_smoothed_ensemble_best_method.npy` (the 3D points),
the previous `<movie>_analysis_smoothed.h5` (for the camera trigger, frame rate and provenance)
and `source.json`, plus the experiment's `perturbation.json` from the server. Experiments the
server has no declaration for keep the one their previous analysis recorded.

**Same results as the server.** A movie re-analysed on a PC matches one re-analysed on the
cluster, apart from floating-point noise between Windows and Linux: wing and body angles agree
to about 0.00000001°, angular acceleration to about 0.002 °/s² on values of around
100,000 °/s². Frame numbers, labels and invalid frames are identical. To compare a PC-made h5
with a cluster-made one, allow a small tolerance rather than exact equality.

**What a render sends and gets back.** The box h5 holds the camera images as nine channels per
frame — three per camera, of which the renderer reads one. Only those are sent, which turns 185 MB
a movie into about 60 MB, together with `calibration.h5`, `prescan_cam_validity.npz` and the
movie's own analysis h5. The cluster reprojects the h5's own 3D points onto the images and encodes
the mp4, which comes back at 40–125 MB depending on the movie's length. Nothing about the round
depends on where the movie's data lived when it was predicted: the member config the cluster reads
is written fresh, naming the copies that were just uploaded.

**How a video's age is known.** Each render leaves a `video.json` beside the mp4 recording the
fingerprint of the 3D points it was drawn from. A later run compares that with the fingerprint in
the analysis h5, so it can tell whether a video is still right **without the box h5 and without
asking the cluster**. That is what lets a run skip rendering a movie whose points never moved.

**What a realign round sends and gets back.** Per flagged movie it uploads each model's
`points_3D_all.npy` and its small config, plus the two current ensemble point files for the
before/after comparison — 10–20 MB by the movie's length, and nothing else: no video, no source movie, no h5. What
comes back is the re-combined points and the model-selection files, about 10 MB compressed. The
round is deleted from the cluster when the files are safely home; `--keep-on-server` leaves it
there. Its state lives in `C:\pose-reanalysis\realign_jobs\<round>.json`, which is what lets a
round be picked up again.

**Nothing runs on the gateway.** `moriah-gw-01` is a login gateway, not a workplace, so the tool
never does anything there: every command it sends — fetching the declarations, checking a file,
receiving an upload, downloading the code, running the repair helper — is wrapped in `srun`, and
slurm runs it on whichever compute node is free. The heavy realignment itself is a separate array
job on top of that. This is why a step can pause with `queued and waiting for resources`, and why
the short questions a repair round asks are spaced a few minutes apart: each one is a small job of
its own. If a node comes up without the lab filesystem mounted, the command waits up to a minute
for it and the short questions are simply handed to slurm again.

**Settings.** They live in `C:\pose-reanalysis\local_reanalysis_settings.json`:

| setting | meaning |
|---|---|
| `server_user` | your username on the server |
| `server_host` | the server address |
| `server_project` | the lab's copy of the project (code and declarations) |
| `upload_to` | where uploads go (empty = `<server_project>/collected_h5`) |
| `upload` | whether this PC publishes to the server at all, and so whether it may repair movies on the cluster. Setup sets it to `false` for an account that cannot write `upload_to` |
| `collected_h5` | where the PC keeps collected files (empty = `C:\pose-reanalysis\collected_h5`) |
| `jobs` | movies at once (0 = half the processor threads; each movie needs about 1.5 GB of memory) |
| `dataset_root` | the folder holding each experiment's source data — the box h5 files and `calibration.h5`; several folders separated by `;`. Rendering needs it (and `predict.bat`, to name experiments); empty means the tool tries only the paths recorded when the movie was predicted |
| `srun_flags` | what the cluster's scheduler is asked for when it runs a command for this PC. Emptying it would run commands on the login gateway instead, which the lab does not allow |
| `predict_output`, `predict_config`, `upload_chunk_mb`, `fetch_chunk_mb`, `server_reserve_gb` | used by `predict.bat`; see [LOCAL_PREDICT.md](LOCAL_PREDICT.md) |

**The same steps on the cluster.**

```bash
.env/bin/python code/reanalyse_movies.py predict_output/<experiment> --only-stale --jobs 8
.env/bin/python code/collect_analysis_h5.py predict_output/<experiment> collected_h5
```

`code/local_reanalysis.py` runs these two for you on a PC, plus the declaration download and the
upload (through `code/local_reanalysis_server.py` on the server). The repair step is
`code/realign_ensemble.py`, which on the cluster is run directly over movies in `predict_output`:

```bash
.env/bin/python code/realign_ensemble.py --list <manifest> --dry-run
sbatch --array=1-$(wc -l < <manifest>) sbatch_files/round_array.sh <manifest> \
    code/realign_ensemble.py --no-reanalyse
```

**Using the collected files downstream.** The upload stops at `collected_h5`. Copying files into
another project is a deliberate, separate step. Keep that project's previous files, for example
`rsync -rc --backup --backup-dir=<project>/data/superseded_<date>/<experiment> collected_h5/<experiment>/ <project>/data/unprocessed_data/<experiment>/`.
Always compare by checksum (`-c`), never `--size-only`, since a re-analysed h5 can have exactly
the old file's size. Files analysed before 2026-09-12 held **nose-up** pitch; files from this
tool hold nose-down (their `pitch_convention` dataset says so).

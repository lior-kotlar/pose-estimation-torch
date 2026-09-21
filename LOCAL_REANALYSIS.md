# Re-analysing predicted movies on your own PC

Use this when movies were **already predicted** (they have 3D points), the prediction folders
are on a disk at your PC, and you want their flight data rebuilt with the current code, without
uploading the movies or predicting again.

For every movie, the tool rewrites the analysis h5, the CSV, the wing/body plots, the flight
viewer and the two plotly pages, right inside the movie's folder. It then gathers all the
`*_analysis_smoothed.h5` files and uploads them to the lab server.

You do the setup once. After that, a run is: **drag a folder onto `reanalyse.bat`**.

---

## What you need

- A Windows 10 or 11 PC.
- The predicted movie folders on a disk. The folder you give can be one experiment, or a folder
  holding many experiments, at any depth.
- An account on the lab server (`moriah-gw-01.cs.huji.ac.il`).

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
ssh YOUR_USERNAME@moriah-gw-01.cs.huji.ac.il "cd /cs/labs/tsevi/lior.kotlar/pose-estimation-torch && git -c 'safe.directory=*' archive --format=tar HEAD code local_reanalysis requirements-analysis.txt LOCAL_REANALYSIS.md" > C:\pose-reanalysis\download.tar
tar -xf C:\pose-reanalysis\download.tar -C C:\pose-reanalysis
del C:\pose-reanalysis\download.tar
```

The first time you connect, ssh asks `Are you sure you want to continue connecting`: type
`yes`. Then type your server password. Nothing shows while you type; that's normal.

Afterwards `C:\pose-reanalysis` contains `code`, `local_reanalysis`, `LOCAL_REANALYSIS.md` and
`requirements-analysis.txt`.

### 3. Run the setup

Double-click **`C:\pose-reanalysis\local_reanalysis\setup.bat`**. It:

1. creates a private Python environment in `C:\pose-reanalysis\venv`,
2. installs the packages it needs (about 1 GB, a few minutes),
3. asks for your **server username** (Enter keeps the suggested server address),
4. asks whether to log in **without a password** from now on. Answer `y`, then type your server
   password one last time,
5. checks that it can reach the server, and whether this account may publish results to it.

It ends with `Setup finished.` Press any key to close the window.

**Only the pipeline's owner uploads.** If your account cannot write the upload folder on the
server, setup says so and turns uploading off for this PC:

```
This account cannot write to /cs/labs/tsevi/lior.kotlar/pose-estimation-torch/collected_h5,
so this PC will re-analyse and collect movies for itself only -- nothing is uploaded.
```

Everything else works exactly the same: your movies are re-analysed in place and collected into
`C:\pose-reanalysis\collected_h5`, and a run then has five steps instead of six. To hand
results over, give the owner that folder (or the movie folders themselves).

---

## Re-analysing movies

**Drag the folder** onto `C:\pose-reanalysis\local_reanalysis\reanalyse.bat`.
Or double-click `reanalyse.bat` and paste the folder's path when asked.

Tip: right-click `reanalyse.bat` → *Send to* → *Desktop (create shortcut)*. You can then drop
folders on the desktop icon.

A window opens and works through six steps:

| step | what happens |
|---|---|
| 1 finding predicted movies | lists the experiments and how many movies each has |
| 2 downloading declarations | fetches each experiment's `perturbation.json` (pulse and lighting) from the server |
| 3 checking every movie | shows, per movie, whether its trigger and declaration were found and whether it needs redoing |
| 4 re-analysing | redoes the movies that need it, several at a time (about 10–20 s per movie) |
| 5 collecting | copies each movie's analysis h5 into `C:\pose-reanalysis\collected_h5` |
| 6 uploading | sends new or changed h5 files to the server, which checks every file before keeping it (skipped when this PC may not upload) |

It ends with a summary like:

```
=== finished in 1.2 min ===
re-analysed      : done 8
report           : C:\pose-reanalysis\reports\reanalyse_report_20260917_130727.csv
log of this run  : C:\pose-reanalysis\reports\run_20260917_130649.log
collected on PC  : C:\pose-reanalysis\collected_h5
on the server    : moriah-gw-01.cs.huji.ac.il:/cs/labs/tsevi/lior.kotlar/pose-estimation-torch/collected_h5
```

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

### What changes in each movie folder

| file | |
|---|---|
| `<movie>_analysis_smoothed.h5` | the flight data (the file the downstream analysis uses) |
| `<movie>_analysis_smoothed.csv` | the same per-frame data as a table |
| `<movie>_analysis_smoothed_flight_viewer.html` | interactive viewer; open it in a browser |
| `wing_angles.png`, `body_angular_acceleration.png` | the plots |
| `All body data.html`, `movie_html.html` | plotly pages |
| `source.json` | where the movie came from, and which code analysed it |
| `superseded_<date>_<time>\` | **the previous versions of all of the above**; nothing is deleted |

The predictions themselves (3D points, model outputs) and `movie 2D and 3D.mp4` are never
touched.

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

- After the code changes, the next run **redoes every movie**, because what the products contain
  is decided by the code that made them. That is the point of the check: it stops you from
  re-analysing with an old copy and having to do it again later.
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

**Settings.** They live in `C:\pose-reanalysis\local_reanalysis_settings.json`:

| setting | meaning |
|---|---|
| `server_user` | your username on the server |
| `server_host` | the server address |
| `server_project` | the lab's copy of the project (code and declarations) |
| `upload_to` | where uploads go (empty = `<server_project>/collected_h5`) |
| `upload` | whether this PC publishes to the server at all. Setup sets it to `false` for an account that cannot write `upload_to` |
| `collected_h5` | where the PC keeps collected files (empty = `C:\pose-reanalysis\collected_h5`) |
| `jobs` | movies at once (0 = half the processor threads; each movie needs about 1.5 GB of memory) |

**The same steps on the cluster.**

```bash
.env/bin/python code/reanalyse_movies.py predict_output/<experiment> --only-stale --jobs 8
.env/bin/python code/collect_analysis_h5.py predict_output/<experiment> collected_h5
```

`code/local_reanalysis.py` runs these two for you on a PC, plus the declaration download and the
upload (through `code/local_reanalysis_server.py` on the server).

**Using the collected files downstream.** The upload stops at `collected_h5`. Copying files into
another project is a deliberate, separate step. Keep that project's previous files, for example
`rsync -rc --backup --backup-dir=<project>/data/superseded_<date>/<experiment> collected_h5/<experiment>/ <project>/data/unprocessed_data/<experiment>/`.
Always compare by checksum (`-c`), never `--size-only`, since a re-analysed h5 can have exactly
the old file's size. Files analysed before 2026-09-12 held **nose-up** pitch; files from this
tool hold nose-down (their `pitch_convention` dataset says so).

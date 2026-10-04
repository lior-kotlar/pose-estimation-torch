# Predicting raw movies from your PC: step by step

Use this guide when an experiment's raw movies are on a disk at your PC and you want them
predicted. The raw movies are the `mov<N>` folders holding one `*_sparse.mat` file per camera.

You do not copy anything to the cluster yourself. You drag the experiment folder onto
`predict.bat`, and the tool does the rest:

1. it sends the movies to the lab cluster;
2. the cluster prepares and predicts them;
3. the results come back into folders on your PC;
4. the cluster's copy is deleted.

---

## Part A — once per PC

### A1. Set up the re-analysis tool

`predict.bat` is part of the re-analysis tool.

- **If this PC already runs `reanalyse.bat`,** skip to A2: the tool updates itself before every
  run, so `predict.bat` is already there or will be after the next run.
- **If not,** follow "One-time setup" in [LOCAL_REANALYSIS.md](LOCAL_REANALYSIS.md) (steps 1–3,
  about 15 minutes).

When you get to the setup question **"Where do you keep the source datasets"**, give the folder
that holds your experiments, for example `E:\Lior\inference_datasets` (see A3).

Check that the folder `C:\pose-reanalysis\local_reanalysis` now contains `predict.bat` and
`predict_check.bat`. If they are missing, double-click `update.bat` in that folder.

### A2. Choose where the predictions go (optional)

By default the predictions go to `C:\pose-reanalysis\predict_output`, which is created by itself.
To put them somewhere else:

1. Open `C:\pose-reanalysis\local_reanalysis_settings.json` in Notepad.
2. Set the `predict_output` line to your folder. Write every `\` twice, as JSON requires:

   ```json
   "predict_output": "E:\\Lior\\pose_esitmation_predict_output",
   ```

3. Save the file.

### A3. Keep your raw data in this layout

```
E:\Lior\inference_datasets\                     <- the "datasets folder" from setup (A1)
    roni_dark\
        2023_08_07_5ms\                         <- one experiment
            10_8_23_allmovs_easyWandData.mat    <- its easyWand calibration
            perturbation.json                   <- optional: the stimulus declaration
            1to30\                              <- batch folders are fine, but optional
                mov1\  mov1_cam1_sparse.mat  mov1_cam2_sparse.mat  ...
                mov2\  ...
```

- Every movie is a folder named `mov<number>` holding one `*_sparse.mat` per camera: 3 or 4, or 2
  on a 2-camera rig.
- The experiment's **easyWand** `.mat` must be in the experiment folder, a batch folder, or one
  or two folders above it.
- **Folder names** may only use letters, digits, `.`, `-` and `_`. No spaces.
- **Keep experiments inside the datasets folder.** The path below it, here
  `roni_dark/2023_08_07_5ms`, becomes the experiment's name in every result. A folder outside the
  datasets folder still works, but is named `local_only/<folder name>`.

---

## Part B — predicting an experiment

### B1. (Optional) Preview what will be sent

Drag the experiment folder onto `predict_check.bat`. It reads only your PC and sends nothing. It
lists every movie and what will happen to it (see B4), then waits for a key press.

### B2. Start

Drag the experiment folder onto **`C:\pose-reanalysis\local_reanalysis\predict.bat`**. You can
also double-click `predict.bat` and paste the folder's path when it asks.

- You can drag a single experiment, a batch folder, a single movie folder, or a folder holding
  many experiments.
- To make this easier: right-click `predict.bat` → *Send to* → *Desktop (create shortcut)*, then
  drop folders on the desktop icon.

### B3. The first time only: settings for the experiment

The first time the tool sees an experiment, it works out three things and saves them as
`prep.json` in the experiment folder:

1. **Which easyWand calibration** belongs to it.
2. **Which camera films through the mirror**, so that camera gets flipped.
3. **The run name**: the name of the folder the predictions go into.

It decides these from the movies themselves, by checking every easyWand it finds against them,
and shows what it chose:

```
  1. 10_8_23_allmovs_easyWandData.mat     flip cam1     worst camera    3.1 px (next hypothesis 120.4 px)
easyWand: 10_8_23_allmovs_easyWandData.mat (the only easyWand that fits these movies)
mirror camera: cam1 (from the mirror check)
```

**It asks you only in these cases.** Type the answer and press Enter. Pressing Enter alone
accepts the suggestion in `[brackets]`.

| question | what to answer |
|---|---|
| `Which easyWand is this experiment's? (number)` | Two calibrations fit about equally well. Type the number of the right one. For multi-day Roni experiments, that is the **end-of-experiment** easyWand. |
| `Mirror camera to flip (e.g. cam1), or none` | The movies could not tell. Use `cam1` for 2023 Roni data and `cam5` for 2022; Tsory's rig has none. |
| `Which camera films from below?` | 2-camera rigs only: `0` or `1`, in the alphabetical order of the camera file names. |
| `Run name (Enter keeps it)` | The suggested name has no date in it. Type a name containing the date, e.g. `roni_dark_2023_08_07_5ms`. |

Later runs on the same experiment reuse `prep.json` and ask nothing.

### B4. Read the list

Next, the tool shows what will happen to each movie:

```
folder                                       movies  send  done  bad  away  skip
roni_dark\2023_08_07_5ms\1to30                   30    27     0    3     0     0

to predict: 27 movie(s); about 2.4 GB to send and 3.9 GB to bring back
```

| column | meaning |
|---|---|
| **send** | will be predicted now |
| **done** | already predicted in your output folder, and skipped |
| **bad** | the fly is visible for too few frames; never sent (the cluster would refuse it too) |
| **away** | the cluster refused it on an earlier run, with the same `prep.json`; not sent again |
| **skip** | the movie folder has the wrong number of camera files |

It then starts sending straight away. You do not need to confirm anything.

### B5. Wait, or turn the PC off

Once every movie is on the cluster, the window says so:

```
EVERYTHING IS ON THE CLUSTER. You may close this window and turn this PC off: the cluster looks
after the round by itself (slurm job 46295656) and slurm emails you when it is done.
```

**From this point you may close the window, turn the PC off and disconnect.** For example, start
the run in the evening, wait for this line (the upload takes about a minute per few movies), and
go home. A small job on the cluster looks after the round while your PC is off. It retries anything
that fails. When everything is done, slurm sends you an email whose subject names the run:

- **`predictions_<run name>_ready`**: everything that could be predicted is done. Drag the same
  folder onto `predict.bat` to bring the results home.
- **`predictions_<run name>_needs_the_PC`**: a preparation job crashed, or something got stuck.
  Drag the same folder onto `predict.bat`. It sends again whatever needs it, and brings home the
  rest.

**Do not turn the PC off before that line.** Movies that have not been uploaded yet would wait until
you run it again.

If you stay at the PC instead, the window keeps going by itself. The tool checks the cluster every 5 minutes, and writes a line
whenever something has changed (and at least every 10 minutes):

```
  [ 40.1 min] predicted 1, predicting 1; home: 0/2
bringing home 1 movie(s) ...
  u1/mov5: predicted -> E:\...\pose_esitmation_predict_output\roni_dark_2023_08_07_5ms\mov_5_532_1166_ds_3tc_7tj
```

Each movie comes back as soon as it is done. Typical times:

| stage | time |
|---|---|
| checking the movies on your PC | about 10 seconds a movie, first time only |
| sending | about 20 seconds a movie, plus the upload |
| preparing, on the cluster | a few minutes a movie. Only one experiment is prepared at a time, so this can wait for another experiment first |
| predicting, on the cluster | 20 minutes to 3 hours a movie, many movies at once |

**Running it again is always safe.** Drag **the same folder** onto `predict.bat`. It picks up where
the round stopped, and brings home whatever finished in the meantime.

**Cluster maintenance:** when the cluster is down for maintenance, the round simply waits, and
carries on by itself when the cluster is back. The tool knows about announced maintenance windows.
Jobs that would run into one are not started until it is over.

### B6. Finished

The run ends with a summary and waits for a key press:

```
=== 4/4  clearing the round off the cluster ===
the round is gone from the cluster
round predict_LIOR-LAPTOP_20261004_130132: predicted 27
report: C:\pose-reanalysis\reports\predict_report_20261004_150644.csv
```

Any movie that was not predicted is listed below the summary, with the reason.

---

## Part C — after it finishes

### C1. Where everything is

| what | where |
|---|---|
| **the predictions**: one folder per movie, with the 3D points, the analysis h5 and CSV, the plots, the flight viewer and the overlay video | `<output folder>\<run name>\<movie>\` |
| the movie's images, for redrawing its video later (`<movie>_render.h5`) | beside the movie's `.mat` files |
| the raw movie (`<movie>_raw_fr30_skip1.mp4`), if there was none | beside the movie's `.mat` files |
| `calibration.h5`, and `process_report.txt` (what the cluster's preparation printed) | in the folder holding the `mov` folders |
| a table of every movie, with what became of it and why | `C:\pose-reanalysis\reports\predict_report_<time>.csv` |
| everything the window showed | `C:\pose-reanalysis\reports\predict_<time>.log` |

Nothing on your PC is ever deleted. A file that gets replaced is first moved into a
`superseded_<time>\` folder beside it.

### C2. Check the movies

Open each movie's `*_flight_viewer.html` or `movie 2D and 3D.mp4`. Move bad ones into a
`bad_signal\<reason>\` folder inside the run folder, as usual.

### C3. Collect and upload

Drag the **output folder** (or the run folder) onto `reanalyse.bat`. This is the usual
re-analysis step:

- it updates each movie's records to point to its folder on your PC;
- it collects the analysis h5 files and uploads them to `collected_h5` on the server;
- it leaves out the movies you moved to `bad_signal`.

Its first lines say the new movies need "re-analyse". That is expected. It may also say
`unsure about the video of N movie(s)`. That is also expected and harmless: those videos are
correct, and they are left alone.

---

## Part D — I want to …

Some of these need an extra word after the folder. To add one, open **Command Prompt** (Start
menu → type `cmd`) and type the line, with your own folder:

```bat
C:\pose-reanalysis\local_reanalysis\predict.bat "E:\Lior\inference_datasets\roni_dark\2023_08_07_5ms" --repredict
```

| I want to … | do this |
|---|---|
| see what would be sent, without sending | drag the folder onto `predict_check.bat` |
| predict again movies that are already done (e.g. after new models are deployed) | add `--repredict`. The old prediction is kept in `superseded_<time>\` |
| change the easyWand or the mirror camera | add `--redeclare`, or edit `prep.json` in the experiment folder |
| send again movies the cluster refused | fix the cause first (usually the easyWand), then add `--retry-failed`. After `--redeclare` they go again by themselves |
| send the predictions somewhere else, just this once | add `--out "F:\some folder"` |
| stop a round and remove it from the cluster | add `--give-up`. Movies already home stay |
| run without any questions | add `--yes`. It stops instead of asking |

---

## Part E — if something goes wrong

These are handled by the tool itself, with nothing for you to do. They are handled even while your
PC is off, by the small job that looks after the round:

- **A GPU job that fails** is retried once.
- **A preparation job that crashes** is sent again and redone, once.
- **A cluster node that loses the lab disk** is noticed, and the job is resubmitted.
- **A dropped connection** is retried at the next check.
- **A nearly full cluster disk**: the tool always leaves 30 GB free for others. If an
  experiment does not fit, the rest goes in a second round, which starts by itself.

| the window says | what to do |
|---|---|
| `STOPPED: ...` (anything) | Read the line, fix what it says, and drag the same folder onto `predict.bat` again. Nothing is lost. |
| `PREP STOPPED -- REFUSING TO FLIP cam1 ...` | `prep.json` names the wrong mirror camera. Run again with `--redeclare`. |
| a movie `away` with `verify FAIL [123.4, ...]` | The easyWand does not fit those movies. Put the right easyWand in the experiment folder and run with `--redeclare`. |
| `no easyWand .mat was found` | Copy the experiment's `*_easyWandData.mat` into the experiment folder. |
| `the easyWand cannot be chosen from the data` | Run `predict.bat` without `--yes` and pick one when asked. |
| `the folder name '...' holds characters the cluster tools cannot take` | Rename that folder: letters, digits, `.`, `-`, `_` only. |
| a movie `rejected` with `the PC that blanked it worked out frames ...` | Your PC and the cluster disagreed about which frames to build. That means your PC's copy of the tool was out of date. Run again, without `--no-update`. |
| `predicting runs jobs on the cluster in the pipeline owner's account` | This PC's account cannot run cluster jobs. Only `predict_check.bat` works here; ask Lior. |
| `N movie(s) have nothing running for them on the cluster; giving up on them` | Run the same folder again; they will be sent anew. |
| it sits at `waiting` for a long time | Another experiment is being prepared first. Only one runs at a time. It continues by itself. |

---

## Part F — how it works, briefly

**What is sent.** The cluster's preparation builds only the longest stretch of a movie in which
every camera sees one whole fly, plus 7 frames on each side. Your PC runs the same check with
the same code. For each camera it sends:

- those frames;
- the 100 frames the cluster's mirror check reads;

and it replaces every other frame with an empty frame. The movie keeps its length, its frame
numbering and its name, but usually less than half its size goes over the network. Movies with
too few usable frames are not sent at all.

**Why the results are the same as uploading by hand.**

- Before anything is sent, your PC checks every cut-down movie.
- The cluster refuses to build a movie from frames that were left out (`TRIM_MISMATCH`).
- A comparison of full and cut-down copies of three movies (a 3-camera Tsory movie, a 4-camera
  Shalev movie and a 4-camera Amitai movie) gave identical results at every step of preparation.

The one difference is the **raw movie**: made from a cut-down movie, it shows only the predicted
stretch. So a raw movie already beside your files is never replaced.

**On the cluster** each run is one folder, `realign_jobs/predict_<PC>_<time>/`, inside the
project. Everything in it is deleted once your PC has the results. A small CPU job named
`keep_predictions_<run name>` looks after the round. It checks every 5 minutes, retries what
failed, and sends the email at the end. If it runs out of time (6 days, or the start of a
maintenance window), it hands over to a new one. While your PC is on and watching too, the PC
leaves the retrying to it. Preparations run one at a time
across all runs, because MATLAB builds slow each other down. As with the rest of the tool,
nothing runs on the login gateway.

### Reference

**`prep.json`** (in the experiment folder):

| field | meaning |
|---|---|
| `easywand` | the calibration file, relative to the experiment folder |
| `cam` | the mirror camera to flip, or `none` |
| `num_cams`, `bottom_cam`, `prep_args` | the camera count. A 2-camera rig also gives the camera filming from below |
| `run_name` | the folder the predictions go into |
| `prescan` | the thresholds that decide which frames are usable. Your PC and the cluster both use these |
| `mirror_check`, `declared_how` | how each choice was made, for the record |

**Settings** (`C:\pose-reanalysis\local_reanalysis_settings.json`):

| setting | meaning |
|---|---|
| `predict_output` | where predictions go (empty: `C:\pose-reanalysis\predict_output`) |
| `dataset_root` | the datasets folder from setup (A3) |
| `predict_config` | which models the cluster uses (default `config1.json`: the deployed ones) |
| `upload_chunk_mb`, `fetch_chunk_mb` | how much goes over one connection (default 2000 MB) |
| `server_reserve_gb` | free space always left on the cluster's disk (default 30) |
| `email_when_done` | `true` (default): email the owner when a round is done; `false`: no email |

**All options** of `predict.bat`: `--check`, `--out FOLDER`, `--redeclare`, `--repredict`,
`--retry-failed`, `--yes`, `--give-up`, `--restart` (start a new round instead of continuing),
`--at-once N` (GPU jobs at a time, default 32), `--poll-seconds N` (default 300),
`--keep-on-server` (keep the cluster copy, for debugging), `--no-update`.

**The same thing by hand, on the cluster**, is described in [PIPELINE.md](PIPELINE.md). The
programs behind this guide are `code/local_predict.py` (on the PC), `code/predict_prep.py`
(checking the movies), `code/sparse_trim.py` (cutting them down) and
`code/local_predict_server.py` (the cluster side).

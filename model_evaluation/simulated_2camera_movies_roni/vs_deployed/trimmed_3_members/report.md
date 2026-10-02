# Simulated 2-camera rig vs the 4-camera reference: roni

Each (bottom, side) pair was cut from the same frames as the all_cams reference and predicted as a 2-camera movie. Distances are to the 4-camera ensemble `sim_roni_all_cams_deployed`, which is not ground truth (about 0.09 mm itself on held-out frames).

## 3D distance to the reference, all movies pooled (mm)

| pair | frames | all median | all p95 | % > 0.5 mm | left wing p95 | right wing p95 | hinges p95 | head & tail p95 | swapped frames |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cam1_cam2 | 7547 | 0.070 | 0.218 | 0.6 | 0.208 | 0.191 | 0.241 | 0.346 | 0 (0.0%) |
| cam1_cam3 | 7547 | 0.071 | 0.195 | 0.3 | 0.175 | 0.187 | 0.249 | 0.221 | 0 (0.0%) |
| cam1_cam4 | 7547 | 0.074 | 0.220 | 0.7 | 0.189 | 0.188 | 0.225 | 0.293 | 0 (0.0%) |

## Angle differences to the reference, all movies pooled (degrees, median / p95)

| pair | phi L | theta L | psi L | phi R | theta R | psi R | yaw | pitch | roll |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cam1_cam2 | 2.0 / 13.3 | 2.2 / 8.0 | 5.2 / 18.2 | 1.7 / 12.0 | 2.4 / 11.2 | 5.1 / 16.8 | 0.6 / 11.0 | 0.7 / 4.7 | 0.5 / 2.0 |
| cam1_cam3 | 1.7 / 5.8 | 2.6 / 9.8 | 4.5 / 17.7 | 1.7 / 7.4 | 2.2 / 8.6 | 4.9 / 17.1 | 0.6 / 4.9 | 0.9 / 5.7 | 0.2 / 1.5 |
| cam1_cam4 | 2.3 / 9.1 | 2.7 / 10.3 | 5.2 / 17.8 | 2.1 / 8.7 | 2.6 / 9.9 | 5.0 / 17.9 | 1.3 / 7.0 | 1.9 / 8.7 | 0.5 / 3.4 |

## Per movie

| movie | pair | frames | all median mm | all p95 mm | swapped | NaN frames | rigidity pair | rigidity reference | body mm (ref) | body off % | axis angle deg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mov1 | cam1_cam2 | 2385 | 0.073 | 0.249 | 0 | 0 | 55.0 | 28.2 | 2.45 (2.47) | -0.9 | 45 |
| mov1 | cam1_cam3 | 2385 | 0.067 | 0.203 | 0 | 0 | 35.2 | 28.2 | 2.48 (2.47) | +0.3 | 31 |
| mov1 | cam1_cam4 | 2385 | 0.086 | 0.251 | 0 | 0 | 49.5 | 28.2 | 2.22 (2.47) | -10.2 | 12 |
| mov12 | cam1_cam2 | 1740 | 0.064 | 0.152 | 0 | 0 | 34.3 | 23.0 | 2.58 (2.53) | +1.9 | 37 |
| mov12 | cam1_cam3 | 1740 | 0.062 | 0.150 | 0 | 0 | 37.0 | 23.0 | 2.54 (2.53) | +0.6 | 45 |
| mov12 | cam1_cam4 | 1740 | 0.056 | 0.143 | 0 | 0 | 37.4 | 23.0 | 2.52 (2.53) | -0.4 | 4 |
| mov13 | cam1_cam2 | 738 | 0.085 | 0.385 | 0 | 0 | 71.5 | 31.7 | 2.14 (2.62) | -18.4 | 15 |
| mov13 | cam1_cam3 | 738 | 0.074 | 0.222 | 0 | 0 | 52.8 | 31.7 | 2.57 (2.62) | -1.7 | 28 |
| mov13 | cam1_cam4 | 738 | 0.109 | 0.344 | 0 | 0 | 79.5 | 31.7 | 2.40 (2.62) | -8.2 | 47 |
| mov5 | cam1_cam2 | 2684 | 0.068 | 0.196 | 0 | 0 | 38.9 | 27.9 | 2.58 (2.59) | -0.2 | 5 |
| mov5 | cam1_cam3 | 2684 | 0.079 | 0.213 | 0 | 0 | 50.5 | 27.9 | 2.54 (2.59) | -2.0 | 39 |
| mov5 | cam1_cam4 | 2684 | 0.070 | 0.172 | 0 | 0 | 49.2 | 27.9 | 2.71 (2.59) | +4.7 | 32 |

Rigidity is the pipeline's own score (mean std of the wing edge lengths, µm; lower is steadier), which the ensemble selector minimises -- necessary, not sufficient. Body: the fly's median tail-to-head length, against the reference's for the same movie (check_body_length.py flags beyond 6 %). Axis angle: median angle between the body axis and the plane through the two cameras.

## Does the ensemble beat its members? (mm, mean over joints, median / p95, all movies pooled)

Each member's own smoothed 3D points against the reference, and the ensemble's, both before the analysis step.

| | cam1_cam2 | cam1_cam3 | cam1_cam4 |
|---|---:|---:|---:|
| ensemble | 0.082 / 0.154 | 0.082 / 0.143 | 0.085 / 0.156 |
| all_cams_2cam_dil3 | 0.089 / 0.168 | 0.087 / 0.174 | 0.091 / 0.168 |
| all_cams_2cam_jsd | 0.090 / 0.176 | 0.093 / 0.154 | 0.094 / 0.181 |
| per_cam_small_dil3 | 0.110 / 0.190 | 0.105 / 0.198 | 0.114 / 0.201 |

## Is the body worst when its axis lies near the cameras' plane?

Angle between the body axis (from the reference) and the plane through the body and the pair's two cameras. Head/tail: mean 3D error of the two body points; length: |error| of the body length. All pairs and movies pooled.

| axis angle | share of frames | head/tail median mm | head/tail p95 mm | length error median mm | length error p95 mm |
|---|---:|---:|---:|---:|---:|
| 0-10 deg | 18.6% | 0.024 | 0.208 | 0.013 | 0.286 |
| 10-20 deg | 14.2% | 0.196 | 0.322 | 0.210 | 0.419 |
| 20-40 deg | 37.4% | 0.060 | 0.173 | 0.047 | 0.170 |
| 40-90 deg | 29.8% | 0.043 | 0.253 | 0.026 | 0.218 |

## The pairs against each other, all movies pooled (mm, all joints)

| pairs | median | p95 |
|---|---:|---:|
| cam1_cam2 vs cam1_cam3 | 0.091 | 0.263 |
| cam1_cam2 vs cam1_cam4 | 0.099 | 0.285 |
| cam1_cam3 vs cam1_cam4 | 0.091 | 0.268 |

## Which members the ensemble picked (share of frames, averaged over movies and joint groups)

| subset | all_cams_2cam_dil3 | all_cams_2cam_jsd | all_cams_dec18 | all_cams_dil3 | all_cams_maxpool | per_cam_dec18 | per_cam_dil3 | per_cam_jsd | per_cam_small_dil3 | per_cam_unet |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cam1_cam2 | 0.71 | 0.78 | – | – | – | – | – | – | 0.35 | – |
| cam1_cam3 | 0.73 | 0.75 | – | – | – | – | – | – | 0.29 | – |
| cam1_cam4 | 0.79 | 0.71 | – | – | – | – | – | – | 0.31 | – |
| all_cams | – | – | 0.24 | 0.25 | 0.22 | 0.45 | 0.53 | 0.47 | – | 0.22 |

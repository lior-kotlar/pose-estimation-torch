# Simulated 2-camera rig vs the 4-camera reference: roni

Each (bottom, side) pair was cut from the same frames as the all_cams reference and predicted as a 2-camera movie. Distances are to the 4-camera ensemble `sim_roni_all_cams_deployed`, which is not ground truth (about 0.09 mm itself on held-out frames).

## 3D distance to the reference, all movies pooled (mm)

| pair | frames | all median | all p95 | % > 0.5 mm | left wing p95 | right wing p95 | hinges p95 | head & tail p95 | swapped frames |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cam1_cam2 | 7547 | 0.069 | 0.212 | 0.5 | 0.204 | 0.192 | 0.242 | 0.265 | 0 (0.0%) |
| cam1_cam3 | 7547 | 0.070 | 0.190 | 0.2 | 0.177 | 0.185 | 0.241 | 0.164 | 0 (0.0%) |
| cam1_cam4 | 7547 | 0.071 | 0.207 | 0.7 | 0.184 | 0.188 | 0.215 | 0.244 | 0 (0.0%) |

## Angle differences to the reference, all movies pooled (degrees, median / p95)

| pair | phi L | theta L | psi L | phi R | theta R | psi R | yaw | pitch | roll |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cam1_cam2 | 1.9 / 8.6 | 2.2 / 8.2 | 4.9 / 18.1 | 1.7 / 8.4 | 2.5 / 11.1 | 5.0 / 17.0 | 0.6 / 7.5 | 0.7 / 5.1 | 0.6 / 2.2 |
| cam1_cam3 | 1.6 / 5.8 | 2.6 / 10.0 | 4.5 / 16.8 | 1.7 / 7.0 | 2.3 / 8.1 | 4.7 / 17.1 | 0.7 / 4.9 | 1.0 / 3.9 | 0.3 / 1.4 |
| cam1_cam4 | 2.0 / 7.5 | 2.4 / 9.1 | 4.6 / 16.0 | 1.8 / 7.6 | 2.4 / 9.0 | 4.5 / 16.5 | 1.0 / 6.2 | 1.9 / 6.5 | 0.3 / 4.1 |

## Per movie

| movie | pair | frames | all median mm | all p95 mm | swapped | NaN frames | rigidity pair | rigidity reference | body mm (ref) | body off % | axis angle deg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mov1 | cam1_cam2 | 2385 | 0.073 | 0.249 | 0 | 0 | 52.5 | 28.2 | 2.45 (2.47) | -0.9 | 45 |
| mov1 | cam1_cam3 | 2385 | 0.067 | 0.202 | 0 | 0 | 35.6 | 28.2 | 2.48 (2.47) | +0.2 | 31 |
| mov1 | cam1_cam4 | 2385 | 0.083 | 0.229 | 0 | 0 | 47.0 | 28.2 | 2.26 (2.47) | -8.6 | 12 |
| mov12 | cam1_cam2 | 1740 | 0.061 | 0.145 | 0 | 0 | 33.4 | 23.0 | 2.55 (2.53) | +0.9 | 37 |
| mov12 | cam1_cam3 | 1740 | 0.060 | 0.149 | 0 | 0 | 35.6 | 23.0 | 2.51 (2.53) | -0.7 | 45 |
| mov12 | cam1_cam4 | 1740 | 0.056 | 0.142 | 0 | 0 | 37.3 | 23.0 | 2.49 (2.53) | -1.4 | 4 |
| mov13 | cam1_cam2 | 738 | 0.082 | 0.294 | 0 | 0 | 62.8 | 31.7 | 2.27 (2.62) | -13.2 | 15 |
| mov13 | cam1_cam3 | 738 | 0.072 | 0.201 | 0 | 0 | 44.5 | 31.7 | 2.55 (2.62) | -2.6 | 28 |
| mov13 | cam1_cam4 | 738 | 0.103 | 0.302 | 0 | 0 | 61.7 | 31.7 | 2.51 (2.62) | -4.0 | 47 |
| mov5 | cam1_cam2 | 2684 | 0.068 | 0.199 | 0 | 0 | 39.0 | 27.9 | 2.58 (2.59) | -0.4 | 5 |
| mov5 | cam1_cam3 | 2684 | 0.077 | 0.205 | 0 | 0 | 47.5 | 27.9 | 2.56 (2.59) | -1.2 | 39 |
| mov5 | cam1_cam4 | 2684 | 0.067 | 0.169 | 0 | 0 | 45.6 | 27.9 | 2.69 (2.59) | +3.7 | 32 |

Rigidity is the pipeline's own score (mean std of the wing edge lengths, µm; lower is steadier), which the ensemble selector minimises -- necessary, not sufficient. Body: the fly's median tail-to-head length, against the reference's for the same movie (check_body_length.py flags beyond 6 %). Axis angle: median angle between the body axis and the plane through the two cameras.

## Does the ensemble beat its members? (mm, mean over joints, median / p95, all movies pooled)

Each member's own smoothed 3D points against the reference, and the ensemble's, both before the analysis step.

| | cam1_cam2 | cam1_cam3 | cam1_cam4 |
|---|---:|---:|---:|
| ensemble | 0.082 / 0.139 | 0.080 / 0.134 | 0.081 / 0.146 |
| all_cams_2cam_dil3 | 0.089 / 0.168 | 0.087 / 0.174 | 0.091 / 0.168 |
| all_cams_2cam_jsd | 0.090 / 0.176 | 0.093 / 0.154 | 0.094 / 0.181 |
| all_cams_2cam_maxfusion | 0.100 / 0.172 | 0.104 / 0.181 | 0.099 / 0.194 |
| per_cam_small_dil3 | 0.110 / 0.190 | 0.105 / 0.198 | 0.114 / 0.201 |
| per_cam_unet | 0.137 / 0.312 | 0.128 / 0.301 | 0.130 / 0.292 |

## Is the body worst when its axis lies near the cameras' plane?

Angle between the body axis (from the reference) and the plane through the body and the pair's two cameras. Head/tail: mean 3D error of the two body points; length: |error| of the body length. All pairs and movies pooled.

| axis angle | share of frames | head/tail median mm | head/tail p95 mm | length error median mm | length error p95 mm |
|---|---:|---:|---:|---:|---:|
| 0-10 deg | 18.6% | 0.020 | 0.275 | 0.028 | 0.438 |
| 10-20 deg | 14.2% | 0.183 | 0.209 | 0.191 | 0.265 |
| 20-40 deg | 37.4% | 0.059 | 0.141 | 0.028 | 0.153 |
| 40-90 deg | 29.8% | 0.043 | 0.186 | 0.023 | 0.101 |

## The pairs against each other, all movies pooled (mm, all joints)

| pairs | median | p95 |
|---|---:|---:|
| cam1_cam2 vs cam1_cam3 | 0.092 | 0.263 |
| cam1_cam2 vs cam1_cam4 | 0.099 | 0.275 |
| cam1_cam3 vs cam1_cam4 | 0.091 | 0.253 |

## Which members the ensemble picked (share of frames, averaged over movies and joint groups)

| subset | all_cams_2cam_dil3 | all_cams_2cam_jsd | all_cams_2cam_maxfusion | all_cams_dec18 | all_cams_dil3 | all_cams_maxpool | per_cam_dec18 | per_cam_dil3 | per_cam_jsd | per_cam_small_dil3 | per_cam_unet |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cam1_cam2 | 0.55 | 0.61 | 0.25 | – | – | – | – | – | – | 0.24 | 0.30 |
| cam1_cam3 | 0.59 | 0.63 | 0.26 | – | – | – | – | – | – | 0.14 | 0.37 |
| cam1_cam4 | 0.63 | 0.62 | 0.25 | – | – | – | – | – | – | 0.16 | 0.28 |
| all_cams | – | – | – | 0.24 | 0.25 | 0.22 | 0.45 | 0.53 | 0.47 | – | 0.22 |

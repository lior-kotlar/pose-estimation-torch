# Simulated 2-camera rig vs the 4-camera reference: roni

Each (bottom, side) pair was cut from the same frames as the all_cams reference and predicted as a 2-camera movie. Distances are to the 4-camera ensemble `sim_roni_all_cams`, which is not ground truth (about 0.09 mm itself on held-out frames).

## 3D distance to the reference, all movies pooled (mm)

| pair | frames | all median | all p95 | % > 0.5 mm | left wing p95 | right wing p95 | hinges p95 | head & tail p95 | swapped frames |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cam1_cam2 | 7547 | 0.061 | 0.200 | 0.5 | 0.202 | 0.179 | 0.230 | 0.200 | 0 (0.0%) |
| cam1_cam3 | 7547 | 0.059 | 0.176 | 0.1 | 0.168 | 0.174 | 0.210 | 0.126 | 0 (0.0%) |
| cam1_cam4 | 7547 | 0.061 | 0.199 | 0.6 | 0.176 | 0.184 | 0.173 | 0.243 | 0 (0.0%) |

## Angle differences to the reference, all movies pooled (degrees, median / p95)

| pair | phi L | theta L | psi L | phi R | theta R | psi R | yaw | pitch | roll |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cam1_cam2 | 1.7 / 8.2 | 2.1 / 8.4 | 4.5 / 17.5 | 1.4 / 8.2 | 2.5 / 10.6 | 4.2 / 14.8 | 0.2 / 5.7 | 0.4 / 4.6 | 0.6 / 2.3 |
| cam1_cam3 | 1.3 / 4.6 | 2.2 / 9.8 | 3.5 / 15.2 | 1.4 / 5.9 | 2.1 / 7.7 | 4.4 / 15.6 | 0.4 / 3.4 | 1.0 / 2.8 | 0.3 / 1.0 |
| cam1_cam4 | 1.7 / 6.7 | 2.5 / 9.0 | 4.3 / 16.2 | 1.5 / 6.6 | 2.4 / 8.8 | 4.0 / 13.9 | 0.8 / 5.5 | 2.3 / 6.6 | 0.3 / 3.9 |

## Per movie

| movie | pair | frames | all median mm | all p95 mm | swapped | NaN frames | rigidity pair | rigidity reference | body mm (ref) | body off % | axis angle deg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mov1 | cam1_cam2 | 2385 | 0.070 | 0.239 | 0 | 0 | 52.5 | 25.2 | 2.45 (2.47) | -0.8 | 45 |
| mov1 | cam1_cam3 | 2385 | 0.051 | 0.173 | 0 | 0 | 35.6 | 25.2 | 2.48 (2.47) | +0.3 | 31 |
| mov1 | cam1_cam4 | 2385 | 0.064 | 0.224 | 0 | 0 | 47.0 | 25.2 | 2.26 (2.47) | -8.4 | 12 |
| mov12 | cam1_cam2 | 1740 | 0.054 | 0.137 | 0 | 0 | 33.4 | 22.8 | 2.55 (2.49) | +2.4 | 37 |
| mov12 | cam1_cam3 | 1740 | 0.052 | 0.146 | 0 | 0 | 35.6 | 22.8 | 2.51 (2.49) | +0.8 | 44 |
| mov12 | cam1_cam4 | 1740 | 0.048 | 0.134 | 0 | 0 | 37.3 | 22.8 | 2.49 (2.49) | +0.1 | 4 |
| mov13 | cam1_cam2 | 738 | 0.074 | 0.289 | 0 | 0 | 62.8 | 29.8 | 2.27 (2.55) | -10.9 | 14 |
| mov13 | cam1_cam3 | 738 | 0.060 | 0.190 | 0 | 0 | 44.5 | 29.8 | 2.55 (2.55) | -0.1 | 28 |
| mov13 | cam1_cam4 | 738 | 0.099 | 0.296 | 0 | 0 | 61.7 | 29.8 | 2.51 (2.55) | -1.5 | 47 |
| mov5 | cam1_cam2 | 2684 | 0.055 | 0.181 | 0 | 0 | 39.0 | 25.2 | 2.58 (2.58) | +0.1 | 5 |
| mov5 | cam1_cam3 | 2684 | 0.070 | 0.193 | 0 | 0 | 47.5 | 25.2 | 2.56 (2.58) | -0.8 | 39 |
| mov5 | cam1_cam4 | 2684 | 0.060 | 0.157 | 0 | 0 | 45.6 | 25.2 | 2.69 (2.58) | +4.2 | 32 |

Rigidity is the pipeline's own score (mean std of the wing edge lengths, µm; lower is steadier), which the ensemble selector minimises -- necessary, not sufficient. Body: the fly's median tail-to-head length, against the reference's for the same movie (check_body_length.py flags beyond 6 %). Axis angle: median angle between the body axis and the plane through the two cameras.

## Does the ensemble beat its members? (mm, mean over joints, median / p95, all movies pooled)

Each member's own smoothed 3D points against the reference, and the ensemble's, both before the analysis step.

| | cam1_cam2 | cam1_cam3 | cam1_cam4 |
|---|---:|---:|---:|
| ensemble | 0.075 / 0.133 | 0.068 / 0.126 | 0.071 / 0.137 |
| all_cams_2cam_dil3 | 0.084 / 0.163 | 0.079 / 0.167 | 0.081 / 0.158 |
| all_cams_2cam_jsd | 0.082 / 0.167 | 0.080 / 0.141 | 0.083 / 0.176 |
| all_cams_2cam_maxfusion | 0.098 / 0.171 | 0.099 / 0.180 | 0.094 / 0.196 |
| per_cam_small_dil3 | 0.108 / 0.184 | 0.100 / 0.195 | 0.110 / 0.196 |
| per_cam_unet | 0.138 / 0.305 | 0.128 / 0.298 | 0.129 / 0.289 |

## Is the body worst when its axis lies near the cameras' plane?

Angle between the body axis (from the reference) and the plane through the body and the pair's two cameras. Head/tail: mean 3D error of the two body points; length: |error| of the body length. All pairs and movies pooled.

| axis angle | share of frames | head/tail median mm | head/tail p95 mm | length error median mm | length error p95 mm |
|---|---:|---:|---:|---:|---:|
| 0-10 deg | 18.3% | 0.007 | 0.274 | 0.005 | 0.381 |
| 10-20 deg | 14.8% | 0.181 | 0.209 | 0.184 | 0.259 |
| 20-40 deg | 38.4% | 0.055 | 0.115 | 0.028 | 0.156 |
| 40-90 deg | 28.5% | 0.041 | 0.139 | 0.019 | 0.071 |

## The pairs against each other, all movies pooled (mm, all joints)

| pairs | median | p95 |
|---|---:|---:|
| cam1_cam2 vs cam1_cam3 | 0.092 | 0.263 |
| cam1_cam2 vs cam1_cam4 | 0.099 | 0.275 |
| cam1_cam3 vs cam1_cam4 | 0.091 | 0.253 |

## Which members the ensemble picked (share of frames, averaged over movies and joint groups)

| subset | all_cams_2cam_dil3 | all_cams_2cam_jsd | all_cams_2cam_maxfusion | all_cams_4cam_dil3 | all_cams_4cam_maxfusion | per_cam_small_dil3 | per_cam_unet |
|---|---:|---:|---:|---:|---:|---:|---:|
| cam1_cam2 | 0.55 | 0.61 | 0.25 | – | – | 0.24 | 0.30 |
| cam1_cam3 | 0.59 | 0.63 | 0.26 | – | – | 0.14 | 0.37 |
| cam1_cam4 | 0.63 | 0.62 | 0.25 | – | – | 0.16 | 0.28 |
| all_cams | 0.54 | 0.46 | 0.41 | 0.15 | 0.26 | 0.25 | 0.29 |

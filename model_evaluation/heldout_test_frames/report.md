# Held-out test frames (41 frames)

COMBINED_MEDIAN is the median of the 3D points of every model in the table: a time-free stand-in for the ensemble, whose selector needs neighbouring frames.

deployed_* are the members in prediction_models/, and COMBINED_MEDIAN_DEPLOYED their stand-in (COMBINED_MEDIAN then covers only the models trained on the split). They predate the split and trained on most of these test frames, so their scores are optimistic: a model that beats them has beaten them for sure, one that loses to them may not really be worse.

## Two-camera rig: bottom + one side camera (pooled over the pairs)

| model | 2D mean px | 2D median px | 2D p95 px | 3D median mm | 3D mean mm | 3D p95 mm |
|---|---:|---:|---:|---:|---:|---:|
| GT_PEAK_FLOOR | 0.00 | 0.00 | 0.00 | 0.082 | 0.085 | 0.135 |
| COMBINED_MEDIAN_DEPLOYED | – | – | – | 0.102 | 0.115 | 0.232 |
| COMBINED_MEDIAN | – | – | – | 0.103 | 0.120 | 0.231 |
| deployed_per_cam_jsd | 1.77 | 1.41 | 4.12 | 0.107 | 0.118 | 0.225 |
| all_cams_2CAM_JSD | 1.91 | 1.41 | 4.47 | 0.109 | 0.126 | 0.250 |
| deployed_per_cam_dec18 | 2.06 | 1.41 | 5.10 | 0.110 | 0.133 | 0.291 |
| all_cams_2CAM_DIL3 | 1.95 | 1.41 | 4.47 | 0.111 | 0.134 | 0.270 |
| deployed_per_cam_dil3 | 2.00 | 1.41 | 5.00 | 0.111 | 0.133 | 0.276 |
| per_cam_JSD_DIL3 | 2.02 | 1.41 | 4.47 | 0.112 | 0.131 | 0.243 |
| all_cams_2CAM_MAXFUSION | 2.24 | 1.41 | 5.66 | 0.114 | 0.147 | 0.338 |
| per_cam_DIL3 | 2.04 | 1.41 | 5.00 | 0.115 | 0.137 | 0.284 |
| per_cam_JSD | 2.07 | 1.41 | 5.00 | 0.115 | 0.141 | 0.260 |
| per_cam_SMALL_DIL3 | 2.46 | 1.41 | 5.84 | 0.117 | 0.158 | 0.348 |
| all_cams_2CAM_BASE | 2.06 | 1.41 | 5.00 | 0.117 | 0.142 | 0.287 |
| per_cam_UNET | 2.43 | 2.00 | 5.39 | 0.117 | 0.156 | 0.314 |
| deployed_per_cam_unet | 2.49 | 2.00 | 5.83 | 0.120 | 0.161 | 0.310 |
| per_cam_BASE | 2.46 | 2.00 | 5.83 | 0.127 | 0.164 | 0.361 |

## Four-camera rig

| model | 2D mean px | 2D median px | 2D p95 px | 3D median mm | 3D mean mm | 3D p95 mm |
|---|---:|---:|---:|---:|---:|---:|
| GT_PEAK_FLOOR | 0.00 | 0.00 | 0.00 | 0.058 | 0.058 | 0.075 |
| COMBINED_MEDIAN | – | – | – | 0.087 | 0.096 | 0.180 |
| deployed_per_cam_jsd | 1.86 | 1.41 | 4.24 | 0.089 | 0.097 | 0.179 |
| all_cams_4CAM_DIL3 | 2.13 | 1.41 | 5.00 | 0.089 | 0.105 | 0.219 |
| all_cams_4CAM_MAXFUSION | 2.17 | 1.41 | 5.39 | 0.091 | 0.107 | 0.204 |
| all_cams_2CAM_JSD | 2.09 | 1.41 | 5.10 | 0.091 | 0.099 | 0.188 |
| COMBINED_MEDIAN_DEPLOYED | – | – | – | 0.093 | 0.103 | 0.201 |
| all_cams_4CAM_BASE | 2.26 | 1.41 | 5.39 | 0.095 | 0.112 | 0.230 |
| per_cam_JSD_DIL3 | 2.22 | 2.00 | 5.00 | 0.096 | 0.106 | 0.195 |
| deployed_per_cam_dec18 | 2.32 | 1.41 | 6.00 | 0.097 | 0.112 | 0.233 |
| all_cams_2CAM_DIL3 | 2.20 | 1.41 | 5.10 | 0.097 | 0.110 | 0.220 |
| per_cam_JSD | 2.24 | 2.00 | 5.00 | 0.098 | 0.108 | 0.209 |
| all_cams_4CAM_JSD | 2.30 | 1.41 | 5.10 | 0.099 | 0.116 | 0.225 |
| deployed_per_cam_dil3 | 2.24 | 1.41 | 5.39 | 0.100 | 0.112 | 0.235 |
| all_cams_2CAM_BASE | 2.30 | 2.00 | 5.39 | 0.102 | 0.116 | 0.228 |
| per_cam_DIL3 | 2.26 | 1.41 | 5.39 | 0.103 | 0.114 | 0.233 |
| all_cams_2CAM_MAXFUSION | 2.59 | 2.00 | 6.71 | 0.104 | 0.121 | 0.281 |
| per_cam_UNET | 2.59 | 2.00 | 6.08 | 0.104 | 0.122 | 0.230 |
| per_cam_SMALL_DIL3 | 2.83 | 2.00 | 7.00 | 0.105 | 0.128 | 0.311 |
| deployed_per_cam_unet | 2.63 | 2.00 | 6.32 | 0.105 | 0.125 | 0.235 |
| deployed_all_cams_dec18 | 2.64 | 2.00 | 6.40 | 0.113 | 0.128 | 0.266 |
| per_cam_BASE | 2.78 | 2.00 | 6.71 | 0.113 | 0.132 | 0.280 |
| deployed_all_cams_maxpool | 3.04 | 2.00 | 8.00 | 0.115 | 0.139 | 0.313 |
| deployed_all_cams_dil3 | 2.76 | 2.00 | 6.08 | 0.118 | 0.137 | 0.286 |

## Side triad (old 3-camera rig)

| model | 2D mean px | 2D median px | 2D p95 px | 3D median mm | 3D mean mm | 3D p95 mm |
|---|---:|---:|---:|---:|---:|---:|
| GT_PEAK_FLOOR | 0.00 | 0.00 | 0.00 | 0.058 | 0.058 | 0.079 |
| all_cams_3CAM_DIL3 | 1.94 | 1.41 | 4.47 | 0.091 | 0.103 | 0.209 |
| COMBINED_MEDIAN | – | – | – | 0.093 | 0.104 | 0.199 |
| deployed_per_cam_jsd | 1.96 | 1.41 | 4.47 | 0.093 | 0.102 | 0.195 |
| COMBINED_MEDIAN_DEPLOYED | – | – | – | 0.094 | 0.107 | 0.229 |
| all_cams_3CAM_MAXFUSION | 2.06 | 1.41 | 5.00 | 0.095 | 0.107 | 0.204 |
| all_cams_3CAM_JSD | 2.12 | 1.41 | 5.00 | 0.099 | 0.114 | 0.211 |
| all_cams_3CAM_BASE | 2.22 | 2.00 | 5.39 | 0.101 | 0.118 | 0.232 |
| per_cam_JSD_DIL3 | 2.41 | 2.00 | 5.39 | 0.102 | 0.121 | 0.220 |
| deployed_all_cams_3cam_concat | 2.35 | 2.00 | 5.83 | 0.103 | 0.121 | 0.260 |
| per_cam_JSD | 2.41 | 2.00 | 5.10 | 0.104 | 0.124 | 0.233 |
| deployed_all_cams_3cam_maxpool | 2.42 | 2.00 | 6.08 | 0.104 | 0.124 | 0.269 |
| deployed_all_cams_3cam_dil3 | 2.41 | 2.00 | 5.67 | 0.105 | 0.127 | 0.255 |
| deployed_per_cam_dec18 | 2.57 | 2.00 | 6.71 | 0.106 | 0.132 | 0.283 |
| per_cam_DIL3 | 2.49 | 2.00 | 6.09 | 0.109 | 0.134 | 0.276 |
| deployed_per_cam_dil3 | 2.47 | 2.00 | 6.08 | 0.109 | 0.131 | 0.277 |
| per_cam_UNET | 2.76 | 2.24 | 6.71 | 0.111 | 0.138 | 0.281 |
| deployed_per_cam_unet | 2.77 | 2.00 | 6.71 | 0.114 | 0.138 | 0.292 |
| per_cam_SMALL_DIL3 | 3.20 | 2.00 | 8.06 | 0.118 | 0.165 | 0.562 |
| per_cam_BASE | 3.09 | 2.24 | 7.81 | 0.126 | 0.162 | 0.375 |

## Validation vs test (informal)

Best-epoch validation error beside the test 2D error in the setting the model was built for. Ball-park only: the 21 validation frames also chose the checkpoint.

| model | val px | test px | test / val |
|---|---:|---:|---:|
| per_cam_BASE | 2.53 | 2.78 | 1.10 |
| per_cam_SMALL_DIL3 | 2.78 | 2.83 | 1.02 |
| per_cam_DIL3 | 2.33 | 2.26 | 0.97 |
| per_cam_JSD | 2.04 | 2.24 | 1.10 |
| per_cam_JSD_DIL3 | 2.09 | 2.22 | 1.06 |
| per_cam_UNET | 2.41 | 2.59 | 1.08 |
| all_cams_4CAM_BASE | 2.03 | 2.26 | 1.12 |
| all_cams_4CAM_DIL3 | 1.96 | 2.13 | 1.09 |
| all_cams_4CAM_MAXFUSION | 2.11 | 2.17 | 1.03 |
| all_cams_4CAM_JSD | 2.15 | 2.30 | 1.07 |
| all_cams_3CAM_BASE | 2.03 | 2.22 | 1.09 |
| all_cams_3CAM_DIL3 | 1.80 | 1.94 | 1.08 |
| all_cams_3CAM_MAXFUSION | 1.84 | 2.06 | 1.12 |
| all_cams_3CAM_JSD | 1.88 | 2.12 | 1.13 |
| all_cams_2CAM_BASE | 1.96 | 2.06 | 1.05 |
| all_cams_2CAM_DIL3 | 1.94 | 1.95 | 1.01 |
| all_cams_2CAM_MAXFUSION | 2.15 | 2.24 | 1.04 |
| all_cams_2CAM_JSD | 1.86 | 1.91 | 1.03 |

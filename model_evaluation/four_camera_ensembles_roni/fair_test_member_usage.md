# Which members the selector picked

Share of frames on which at least one member was picked, averaged over the movies (`ensemble_model_selection_summary.json`, overall).

## deployed (7)

| member | family | frames selected |
|---|---|---:|
| per_cam_dil3 | per-camera | 53% |
| per_cam_jsd | per-camera | 47% |
| per_cam_dec18 | per-camera | 45% |
| all_cams_dil3 | 4-camera | 25% |
| all_cams_dec18 | 4-camera | 24% |
| all_cams_maxpool | 4-camera | 22% |
| per_cam_unet | per-camera | 22% |

## pilot: new with 2-cam (7)

| member | family | frames selected |
|---|---|---:|
| all_cams_2cam_dil3 | 2-camera | 54% |
| all_cams_2cam_jsd | 2-camera | 46% |
| all_cams_2cam_maxfusion | 2-camera | 41% |
| per_cam_unet | per-camera | 29% |
| all_cams_4cam_maxfusion | 4-camera | 26% |
| per_cam_small_dil3 | per-camera | 25% |
| all_cams_4cam_dil3 | 4-camera | 15% |

## A planned (8)

| member | family | frames selected |
|---|---|---:|
| per_cam_jsd_dil3 | per-camera | 46% |
| all_cams_4cam_maxfusion | 4-camera | 41% |
| per_cam_jsd | per-camera | 41% |
| per_cam_dil3 | per-camera | 37% |
| per_cam_unet | per-camera | 30% |
| all_cams_4cam_base | 4-camera | 17% |
| all_cams_4cam_jsd | 4-camera | 16% |
| all_cams_4cam_dil3 | 4-camera | 16% |

## B planned + 2-cam (11)

| member | family | frames selected |
|---|---|---:|
| all_cams_2cam_dil3 | 2-camera | 44% |
| all_cams_2cam_jsd | 2-camera | 35% |
| all_cams_2cam_maxfusion | 2-camera | 34% |
| per_cam_jsd | per-camera | 28% |
| per_cam_jsd_dil3 | per-camera | 26% |
| per_cam_unet | per-camera | 23% |
| all_cams_4cam_maxfusion | 4-camera | 22% |
| per_cam_dil3 | per-camera | 20% |
| all_cams_4cam_jsd | 4-camera | 9% |
| all_cams_4cam_dil3 | 4-camera | 9% |
| all_cams_4cam_base | 4-camera | 8% |

## C same count, 2 swapped for 2-cam (8)

| member | family | frames selected |
|---|---|---:|
| all_cams_2cam_dil3 | 2-camera | 57% |
| all_cams_2cam_jsd | 2-camera | 41% |
| per_cam_jsd | per-camera | 38% |
| per_cam_jsd_dil3 | per-camera | 37% |
| all_cams_4cam_maxfusion | 4-camera | 33% |
| all_cams_4cam_dil3 | 4-camera | 13% |
| all_cams_4cam_jsd | 4-camera | 12% |
| all_cams_4cam_base | 4-camera | 10% |

## D planned + 2CAM_JSD (9)

| member | family | frames selected |
|---|---|---:|
| all_cams_2cam_jsd | 2-camera | 51% |
| per_cam_jsd_dil3 | per-camera | 37% |
| per_cam_dil3 | per-camera | 32% |
| all_cams_4cam_maxfusion | 4-camera | 31% |
| per_cam_jsd | per-camera | 31% |
| per_cam_unet | per-camera | 28% |
| all_cams_4cam_dil3 | 4-camera | 13% |
| all_cams_4cam_base | 4-camera | 12% |
| all_cams_4cam_jsd | 4-camera | 11% |


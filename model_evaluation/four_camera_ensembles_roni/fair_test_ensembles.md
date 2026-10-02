# Four ensembles on the same 4-camera Roni movies

No ground truth on whole movies: the pipeline's own self-consistency scores. Lower is steadier everywhere; body length is the fly's tail-to-head length, whose spread over a flight should be small.

| movie | ensemble | rigidity µm | jitter µm | raw->smooth µm | body mm | body spread mm |
|---|---|---:|---:|---:|---:|---:|
| mov_12_8_1747 | deployed (7) | 23.0 | 15.8 | 29.5 | 2.528 | 0.011 |
| mov_12_8_1747 | pilot: new with 2-cam (7) | 22.8 | 15.6 | 27.2 | 2.490 | 0.015 |
| mov_12_8_1747 | A planned (8) | 23.8 | 15.4 | 29.4 | 2.501 | 0.018 |
| mov_12_8_1747 | B planned + 2-cam (11) | 22.3 | 15.5 | 27.9 | 2.514 | 0.022 |
| mov_12_8_1747 | C same count, 2 swapped for 2-cam (8) | 22.2 | 15.6 | 28.0 | 2.528 | 0.016 |
| mov_12_8_1747 | D planned + 2CAM_JSD (9) | 23.2 | 15.4 | 28.9 | 2.510 | 0.023 |
| mov_13_229_966 | deployed (7) | 31.7 | 19.7 | 40.0 | 2.619 | 0.031 |
| mov_13_229_966 | pilot: new with 2-cam (7) | 29.8 | 20.1 | 39.3 | 2.553 | 0.032 |
| mov_13_229_966 | A planned (8) | 33.3 | 19.4 | 44.1 | 2.589 | 0.033 |
| mov_13_229_966 | B planned + 2-cam (11) | 28.7 | 19.4 | 37.9 | 2.590 | 0.028 |
| mov_13_229_966 | C same count, 2 swapped for 2-cam (8) | 29.7 | 19.5 | 39.2 | 2.616 | 0.021 |
| mov_13_229_966 | D planned + 2CAM_JSD (9) | 30.9 | 19.2 | 41.3 | 2.589 | 0.033 |
| mov_1_8_2392 | deployed (7) | 28.2 | 17.5 | 34.4 | 2.474 | 0.025 |
| mov_1_8_2392 | pilot: new with 2-cam (7) | 25.2 | 16.7 | 30.0 | 2.471 | 0.014 |
| mov_1_8_2392 | A planned (8) | 26.3 | 16.5 | 33.3 | 2.485 | 0.018 |
| mov_1_8_2392 | B planned + 2-cam (11) | 24.7 | 16.6 | 31.6 | 2.483 | 0.015 |
| mov_1_8_2392 | C same count, 2 swapped for 2-cam (8) | 25.1 | 16.6 | 31.7 | 2.488 | 0.014 |
| mov_1_8_2392 | D planned + 2CAM_JSD (9) | 24.4 | 16.1 | 32.3 | 2.485 | 0.017 |
| mov_5_8_2691 | deployed (7) | 27.9 | 17.2 | 34.6 | 2.589 | 0.014 |
| mov_5_8_2691 | pilot: new with 2-cam (7) | 25.2 | 16.9 | 29.9 | 2.578 | 0.013 |
| mov_5_8_2691 | A planned (8) | 27.8 | 16.9 | 33.4 | 2.586 | 0.012 |
| mov_5_8_2691 | B planned + 2-cam (11) | 24.7 | 16.8 | 30.4 | 2.585 | 0.011 |
| mov_5_8_2691 | C same count, 2 swapped for 2-cam (8) | 24.9 | 16.9 | 31.7 | 2.589 | 0.011 |
| mov_5_8_2691 | D planned + 2CAM_JSD (9) | 25.4 | 16.7 | 31.8 | 2.587 | 0.012 |

## Means over the movies

| ensemble | rigidity µm | jitter µm | raw->smooth µm | body spread mm |
|---|---:|---:|---:|---:|
| deployed (7) | 27.7 | 17.6 | 34.6 | 0.020 |
| pilot: new with 2-cam (7) | 25.8 | 17.3 | 31.6 | 0.018 |
| A planned (8) | 27.8 | 17.0 | 35.0 | 0.020 |
| B planned + 2-cam (11) | 25.1 | 17.1 | 31.9 | 0.019 |
| C same count, 2 swapped for 2-cam (8) | 25.5 | 17.1 | 32.7 | 0.015 |
| D planned + 2CAM_JSD (9) | 26.0 | 16.9 | 33.6 | 0.021 |

## How far apart they land (mm, all joints, median / p95, movies pooled)

| | deployed (7) | pilot: new with 2-cam (7) | A planned (8) | B planned + 2-cam (11) | C same count, 2 swapped for 2-cam (8) | D planned + 2CAM_JSD (9) |
|---|---:|---:|---:|---:|---:|---:|
| deployed (7) | – | 0.041 / 0.108 | 0.042 / 0.111 | 0.040 / 0.105 | 0.043 / 0.114 | 0.043 / 0.113 |
| pilot: new with 2-cam (7) | 0.041 / 0.108 | – | 0.033 / 0.087 | 0.020 / 0.076 | 0.025 / 0.087 | 0.028 / 0.080 |
| A planned (8) | 0.042 / 0.111 | 0.033 / 0.087 | – | 0.028 / 0.081 | 0.029 / 0.087 | 0.018 / 0.065 |
| B planned + 2-cam (11) | 0.040 / 0.105 | 0.020 / 0.076 | 0.028 / 0.081 | – | 0.017 / 0.072 | 0.021 / 0.074 |
| C same count, 2 swapped for 2-cam (8) | 0.043 / 0.114 | 0.025 / 0.087 | 0.029 / 0.087 | 0.017 / 0.072 | – | 0.023 / 0.077 |
| D planned + 2CAM_JSD (9) | 0.043 / 0.113 | 0.028 / 0.080 | 0.018 / 0.065 | 0.021 / 0.074 | 0.023 / 0.077 | – |

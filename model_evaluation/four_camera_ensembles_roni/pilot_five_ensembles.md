# Four ensembles on the same 4-camera Roni movies

No ground truth on whole movies: the pipeline's own self-consistency scores. Lower is steadier everywhere; body length is the fly's tail-to-head length, whose spread over a flight should be small.

| movie | ensemble | rigidity µm | jitter µm | raw->smooth µm | body mm | body spread mm |
|---|---|---:|---:|---:|---:|---:|
| mov_12_8_1747 | deployed (7: 3x 4cam, 4x per-cam) | 23.0 | 15.8 | 29.5 | 2.528 | 0.011 |
| mov_12_8_1747 | new, with 2-cam members (7) | 22.8 | 15.6 | 27.2 | 2.490 | 0.015 |
| mov_12_8_1747 | new, without 2-cam members (4) | 27.1 | 16.2 | 29.6 | 2.477 | 0.008 |
| mov_12_8_1747 | new, 2x 2cam + 2x 4cam (4) | 24.7 | 16.0 | 28.0 | 2.503 | 0.011 |
| mov_12_8_1747 | new, 4-cam members only (2) | 32.3 | 17.7 | 36.8 | 2.487 | 0.009 |
| mov_13_229_966 | deployed (7: 3x 4cam, 4x per-cam) | 31.7 | 19.7 | 40.0 | 2.619 | 0.031 |
| mov_13_229_966 | new, with 2-cam members (7) | 29.8 | 20.1 | 39.3 | 2.553 | 0.032 |
| mov_13_229_966 | new, without 2-cam members (4) | 34.3 | 20.4 | 45.2 | 2.539 | 0.036 |
| mov_13_229_966 | new, 2x 2cam + 2x 4cam (4) | 32.9 | 20.1 | 42.3 | 2.560 | 0.058 |
| mov_13_229_966 | new, 4-cam members only (2) | 53.1 | 22.9 | 62.6 | 2.420 | 0.150 |
| mov_1_8_2392 | deployed (7: 3x 4cam, 4x per-cam) | 28.2 | 17.5 | 34.4 | 2.474 | 0.025 |
| mov_1_8_2392 | new, with 2-cam members (7) | 25.2 | 16.7 | 30.0 | 2.471 | 0.014 |
| mov_1_8_2392 | new, without 2-cam members (4) | 29.3 | 17.2 | 35.5 | 2.424 | 0.020 |
| mov_1_8_2392 | new, 2x 2cam + 2x 4cam (4) | 26.2 | 16.9 | 31.3 | 2.482 | 0.016 |
| mov_1_8_2392 | new, 4-cam members only (2) | 33.2 | 18.7 | 49.6 | 2.430 | 0.019 |
| mov_5_8_2691 | deployed (7: 3x 4cam, 4x per-cam) | 27.9 | 17.2 | 34.6 | 2.589 | 0.014 |
| mov_5_8_2691 | new, with 2-cam members (7) | 25.2 | 16.9 | 29.9 | 2.578 | 0.013 |
| mov_5_8_2691 | new, without 2-cam members (4) | 30.1 | 17.3 | 33.8 | 2.562 | 0.015 |
| mov_5_8_2691 | new, 2x 2cam + 2x 4cam (4) | 26.2 | 17.3 | 30.3 | 2.583 | 0.011 |
| mov_5_8_2691 | new, 4-cam members only (2) | 35.0 | 18.6 | 42.6 | 2.573 | 0.014 |

## Means over the movies

| ensemble | rigidity µm | jitter µm | raw->smooth µm | body spread mm |
|---|---:|---:|---:|---:|
| deployed (7: 3x 4cam, 4x per-cam) | 27.7 | 17.6 | 34.6 | 0.020 |
| new, with 2-cam members (7) | 25.8 | 17.3 | 31.6 | 0.018 |
| new, without 2-cam members (4) | 30.2 | 17.8 | 36.0 | 0.020 |
| new, 2x 2cam + 2x 4cam (4) | 27.5 | 17.6 | 33.0 | 0.024 |
| new, 4-cam members only (2) | 38.4 | 19.5 | 47.9 | 0.048 |

## How far apart they land (mm, all joints, median / p95, movies pooled)

| | deployed (7: 3x 4cam, 4x per-cam) | new, with 2-cam members (7) | new, without 2-cam members (4) | new, 2x 2cam + 2x 4cam (4) | new, 4-cam members only (2) |
|---|---:|---:|---:|---:|---:|
| deployed (7: 3x 4cam, 4x per-cam) | – | 0.041 / 0.108 | 0.044 / 0.112 | 0.045 / 0.118 | 0.056 / 0.139 |
| new, with 2-cam members (7) | 0.041 / 0.108 | – | 0.032 / 0.091 | 0.022 / 0.089 | 0.046 / 0.129 |
| new, without 2-cam members (4) | 0.044 / 0.112 | 0.032 / 0.091 | – | 0.039 / 0.112 | 0.038 / 0.123 |
| new, 2x 2cam + 2x 4cam (4) | 0.045 / 0.118 | 0.022 / 0.089 | 0.039 / 0.112 | – | 0.044 / 0.125 |
| new, 4-cam members only (2) | 0.056 / 0.139 | 0.046 / 0.129 | 0.038 / 0.123 | 0.044 / 0.125 | – |

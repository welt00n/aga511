# Checks: passes-cpu64

2026-11-22T11:29:45 · Intel(R) Core(TM) Ultra 7 255HX (one core) · code `72b86b63cbf7` · passed

**1 pass = 4 passes, bit for bit**

`python main.py check all -n 1e7`

## What was asked

| option | value |
|---|---|
| photons | 10000000 |
| seed | 42 |
| limb | 1.0 |
| spots | 3 |
| observers | [1.5708, 1.0472, 0.5236, 0.0] |
| size | 200 |
| frames | 150 |
| chunk_size | 1024 |
| check | passes |
| processor | cpu |
| precision | 64 |

## Results

N = 1e+07, 200² pixels, 150 frames, nmu = odd nearest frames/π, observers at 1.5708, 1.0472, 0.5236, 0 rad, 3 spots

| run | photons/s | photons (s) | setup (s) | zero (s) | copy (s) | save (s) | passes | recorded | blocked |
|---|---|---|---|---|---|---|---|---|---|
| 1 pass | 7.57e+06 | 1.321 | 0.039 | 0.005 | 0.000 | 0.288 | 1 | 843339 (8.43%) | 41308 |
| 4 passes of 1 observer | 9.07e+06 | 4.411 | 0.012 | 0.004 | 0.000 | 0.165 | 4 | 843339 (8.43%) | 41308 |

- counts: the same

## Files

- [`run.json`](run.json)

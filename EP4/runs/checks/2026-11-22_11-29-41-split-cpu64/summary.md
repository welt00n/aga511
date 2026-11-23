# Checks: split-cpu64

2026-11-22T11:29:41 · Intel(R) Core(TM) Ultra 7 255HX (one core) · code `72b86b63cbf7` · passed

**two halves = the full run, bit for bit**

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
| check | split |
| processor | cpu |
| precision | 64 |

## Results

N = 1e+07, 200² pixels, 150 frames, nmu = odd nearest frames/π, observers at 1.5708, 1.0472, 0.5236, 0 rad, 3 spots, chunk size 1024

| run | photons/s | photons (s) | setup (s) | zero (s) | copy (s) | save (s) | passes | recorded | blocked |
|---|---|---|---|---|---|---|---|---|---|
| [0, 9766) | 7.46e+06 | 1.341 | 0.039 | 0.004 | 0.000 | 0.160 | 1 | 843339 (8.43%) | 41308 |
| [0, 4883) | 6.43e+06 | 0.778 | 0.040 | 0.004 | 0.000 | 0.133 | 1 | 421877 (8.44%) | 20475 |
| [4883, 9766) | 6.99e+06 | 0.715 | 0.051 | 0.004 | 0.000 | 0.153 | 1 | 421462 (8.43%) | 20833 |

- recorded: 421877 + 421462 = 843339 vs 843339

## Files

- [`run.json`](run.json)

# Checks: gpu-vs-cpu-fp64

2026-11-22T11:29:54 · Intel(R) Core(TM) Ultra 7 255HX (one core); NVIDIA GeForce RTX 5070 Laptop GPU (cc 12.0, 36 SMs) · code `72b86b63cbf7` · passed

**images identical, bit for bit; 39× one CPU core on the photons, 4× on the run**

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
| check | gpu |
| processor | cpu |
| precision | 64 |

## Results

The same seed: the same random numbers, so the same photons.

N = 1e+07, 200² pixels, 150 frames, nmu = odd nearest frames/π, observers at 1.5708, 1.0472, 0.5236, 0 rad, 3 spots

| run | photons/s | photons (s) | setup (s) | zero (s) | copy (s) | save (s) | passes | recorded | blocked |
|---|---|---|---|---|---|---|---|---|---|
| CPU (one core) | 8.11e+06 | 1.234 | 0.054 | 0.004 | 0.000 | 0.114 | 1 | 843339 (8.43%) | 41308 |
| GPU (all SMs) | 3.17e+08 | 0.032 | 0.226 | 0.000 | 0.005 | 0.118 | 1 | 843339 (8.43%) | 41308 |

- speedup, the whole GPU / one CPU core: 39.1× on the photons, 3.7× on the whole run
- counts: the same
- images: identical, bit for bit

## Files

- [`run.json`](run.json)

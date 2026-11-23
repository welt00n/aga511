# Checks: precision-cpu

2026-11-22T11:29:51 · Intel(R) Core(TM) Ultra 7 255HX (one core) · code `72b86b63cbf7` · done

**FP32 moves 1.01e-04 of the photons (noise 3.87e-01); 1.47× the speed of FP64**

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
| check | precision |
| processor | cpu |
| precision | 64 |

## Results

The same seed: the same random numbers, so the same photons.

N = 1e+07, 200² pixels, 150 frames, nmu = odd nearest frames/π, observers at 1.5708, 1.0472, 0.5236, 0 rad, 3 spots

| run | photons/s | photons (s) | setup (s) | zero (s) | copy (s) | save (s) | passes | recorded | blocked |
|---|---|---|---|---|---|---|---|---|---|
| FP64 | 7.89e+06 | 1.268 | 0.047 | 0.003 | 0.000 | 0.115 | 1 | 843339 (8.43%) | 41308 |
| FP32 | 1.16e+07 | 0.862 | 0.054 | 0.005 | 0.000 | 0.144 | 1 | 843339 (8.43%) | 41308 |

- FP32 / FP64 speed: 1.47
- photons that land in another pixel in FP32: 1.01e-04
- Poisson noise per lit pixel (frames summed): 3.87e-01

## Files

- [`run.json`](run.json)

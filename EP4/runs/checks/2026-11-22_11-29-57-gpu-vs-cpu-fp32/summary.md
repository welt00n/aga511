# Checks: gpu-vs-cpu-fp32

2026-11-22T11:29:57 · Intel(R) Core(TM) Ultra 7 255HX (one core); NVIDIA GeForce RTX 5070 Laptop GPU (cc 12.0, 36 SMs) · code `72b86b63cbf7` · done

**images 4 pixels differ; 2.37e-05 of the photons moved; 39× one CPU core on the photons, 1× on the run**

`python main.py check gpu -n 1e6 --precision 32`

## What was asked

| option | value |
|---|---|
| photons | 1000000 |
| seed | 42 |
| limb | 1.0 |
| spots | 3 |
| observers | [1.5708, 1.0472, 0.5236, 0.0] |
| size | 200 |
| frames | 150 |
| chunk_size | 1024 |
| check | gpu |
| processor | cpu |
| precision | 32 |

## Results

The same seed: the same random numbers, so the same photons.

N = 1e+06, 200² pixels, 150 frames, nmu = odd nearest frames/π, observers at 1.5708, 1.0472, 0.5236, 0 rad, 3 spots

| run | photons/s | photons (s) | setup (s) | zero (s) | copy (s) | save (s) | passes | recorded | blocked |
|---|---|---|---|---|---|---|---|---|---|
| CPU (one core) | 1.14e+07 | 0.088 | 0.047 | 0.005 | 0.000 | 0.117 | 1 | 84337 (8.43%) | 4118 |
| GPU (all SMs) | 4.49e+08 | 0.002 | 0.167 | 0.000 | 0.007 | 0.094 | 1 | 84337 (8.43%) | 4118 |

- speedup, the whole GPU / one CPU core: 39.4× on the photons, 1.0× on the whole run
- counts: the same
- images: 4 pixels differ; 2.37e-05 of the photons moved

## Files

- [`run.json`](run.json)

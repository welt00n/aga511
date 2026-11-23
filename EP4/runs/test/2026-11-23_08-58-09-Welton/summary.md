# Test: Welton

2026-11-23T08:58:09 · Intel(R) Core(TM) Ultra 7 255HX (the sequential reference); NVIDIA GeForce RTX 5070 Laptop GPU (cc 12.0, 36 SMs) · code `72b86b63cbf7` · passed

**all 12 tests passed**

`python main.py test --yes`

## What was asked

| option | value |
|---|---|
| photons | 10000000 |
| tests | ['build', 'rng cpu', 'rng gpu', 'run cpu64', 'run cpu32', 'run gpu64', 'run gpu32', 'split cpu64', 'split gpu64', 'passes gpu64', 'gpu = cpu', 'gpu ~ cpu'] |

## Results

| test | proves | result | detail | s |
|---|---|---|---|---|
| build | every engine compiles here | pass | 4 binaries built | 0.0 |
| rng cpu | Philox on the CPU gives the published numbers | pass | the numbers match the reference | 0.0 |
| rng gpu | cuRAND's Philox on the GPU = the CPU's, bit for bit | pass | cuRAND = the CPU's copy | 0.2 |
| run cpu64 | the CPU engine runs (FP64) | pass | 843339 of 1e+07 photons recorded | 1.3 |
| run cpu32 | the CPU engine runs (FP32) | pass | 843339 of 1e+07 photons recorded | 0.9 |
| run gpu64 | the GPU engine runs (FP64) | pass | 843339 of 1e+07 photons recorded | 0.3 |
| run gpu32 | the GPU engine runs (FP32) | pass | 843339 of 1e+07 photons recorded | 0.3 |
| split cpu64 | any split of the chunks gives the same image (CPU) | pass | two halves = the full run, bit for bit | 2.8 |
| split gpu64 | any split of the chunks gives the same image (GPU) | pass | two halves = the full run, bit for bit | 1.1 |
| passes gpu64 | any number of passes gives the same image (GPU) | pass | 1 pass = 4 passes, bit for bit | 0.9 |
| gpu = cpu | the GPU's image = the CPU's, bit for bit (FP64) | pass | the GPU vs the sequential CPU: the images are identical, bit for bit | 1.9 |
| gpu ~ cpu | the GPU and the sequential CPU agree with independent photons (χ²) | pass | χ²/dof = 1.0106: the same physics within noise | 2.3 |

Correctness only: these runs are far below the minimum N, so their times are not measurements. `measure` makes those.

## Files

- [`checks/`](checks/)
- [`run.json`](run.json)
- [`test.json`](test.json)

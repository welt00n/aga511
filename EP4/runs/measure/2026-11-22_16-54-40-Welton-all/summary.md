# Measure: Welton-all

2026-11-22T16:54:40 · NVIDIA GeForce RTX 5070 Laptop GPU (cc 12.0, 36 SMs); Intel(R) Core(TM) Ultra 7 255HX (one core) · code `72b86b63cbf7` · passed

**GPU 6.1e+08/s FP64 (69× one core), 1e+10/s FP32; strong FP64 35.6× on 36 SMs; FP64 fastest at 72 registers (occupancy 0.58); validation passed**

`python main.py measure --seconds 0.3 --repeats 1 --allow-invalid --yes`

## What was asked

| option | value |
|---|---|
| parts | ['validation', 'kernel', 'occupancy', 'devices', 'scaling'] |
| seconds | 0.3 |
| repeats | 1 |
| targets | ['80'] |
| allow_invalid | True |
| photons | {'cpu64': 8000000, 'cpu32': 15000000, 'gpu64': 1000000000, 'gpu32': 15000000000} |

## Results

**The report:** [report.pdf](report.pdf) and [report.ipynb](report.ipynb) (the same, with the code that reads the data).

- `validation`: [`parts/validation.json`](parts/validation.json)
- `kernel`: [`parts/kernel.json`](parts/kernel.json)
- `occupancy`: [`parts/occupancy.json`](parts/occupancy.json)
- `devices`: [`parts/devices.json`](parts/devices.json)
- `scaling`: [`parts/scaling.json`](parts/scaling.json)

## Figures

![amdahl_ceiling_fp32.png](figures/amdahl_ceiling_fp32.png)
![devices.png](figures/devices.png)
![problem_size.png](figures/problem_size.png)
![registers_and_occupancy.png](figures/registers_and_occupancy.png)
![strong_fp32.png](figures/strong_fp32.png)
![strong_fp32_small_n.png](figures/strong_fp32_small_n.png)
![strong_fp64.png](figures/strong_fp64.png)
![strong_vs_weak_fp32.png](figures/strong_vs_weak_fp32.png)
![strong_vs_weak_fp64.png](figures/strong_vs_weak_fp64.png)
![weak_fp32.png](figures/weak_fp32.png)
![weak_fp64.png](figures/weak_fp64.png)

## Files

- [`figures/`](figures/)
- [`machine.json`](machine.json)
- [`parts/`](parts/)
- [`plan.json`](plan.json)
- [`points/`](points/)
- [`progress.log`](progress.log)
- [`report.html`](report.html)
- [`report.ipynb`](report.ipynb)
- [`report.pdf`](report.pdf)
- [`run.json`](run.json)
- [`summary.md`](summary.md)
- [`validation/`](validation/)

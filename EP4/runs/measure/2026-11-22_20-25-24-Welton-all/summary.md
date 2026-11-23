# Measure: Welton-all

2026-11-22T20:25:24 · NVIDIA GeForce RTX 5070 Laptop GPU (cc 12.0, 36 SMs); Intel(R) Core(TM) Ultra 7 255HX (the sequential reference) · code `72b86b63cbf7` · passed

**GPU 6.7e+08/s FP64 (36.3× one SM), 1.1e+10/s FP32 (35.3× one SM); strong FP64 36.3× on 36 SMs; FP64 fastest at 72 registers (occupancy 0.58); validation passed**

`python main.py measure --yes`

## What was asked

| option | value |
|---|---|
| parts | ['validation', 'speed', 'kernel', 'occupancy', 'scaling', 'costs'] |
| seconds | 5.0 |
| repeats | 3 |
| targets | ['80'] |
| allow_invalid | False |
| photons | {'gpu64': 4000000000, 'gpu32': 50000000000} |
| star | {'size': 1024, 'frames': 300} |

## Results

**The report:** [report.pdf](report.pdf) and [report.ipynb](report.ipynb) (the same, with the code that reads the data).

- `validation`: [`parts/validation.json`](parts/validation.json)
- `speed`: [`parts/speed.json`](parts/speed.json)
- `kernel`: [`parts/kernel.json`](parts/kernel.json)
- `occupancy`: [`parts/occupancy.json`](parts/occupancy.json)
- `scaling`: [`parts/scaling.json`](parts/scaling.json)
- `costs`: [`parts/costs.json`](parts/costs.json)

## Figures

![amdahl_ceiling_fp32.png](figures/amdahl_ceiling_fp32.png)
![amdahl_ceiling_fp64.png](figures/amdahl_ceiling_fp64.png)
![costs_compute.png](figures/costs_compute.png)
![costs_large_n.png](figures/costs_large_n.png)
![costs_memory.png](figures/costs_memory.png)
![problem_size.png](figures/problem_size.png)
![registers_and_occupancy.png](figures/registers_and_occupancy.png)
![speed.png](figures/speed.png)
![strong_fp32.png](figures/strong_fp32.png)
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
- [`report.ipynb`](report.ipynb)
- [`report.pdf`](report.pdf)
- [`run.json`](run.json)
- [`summary.md`](summary.md)
- [`validation/`](validation/)

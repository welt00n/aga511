# EP3: Buffon's needle on the GPU

EP1's Monte Carlo estimate of π, accelerated with CUDA, with Amdahl's law measured over the GPU's SMs.

## Requirements

- NVIDIA GPU, CUDA toolkit (`nvcc`)
- Python 3 with numpy, pandas, matplotlib

## Engine: one experiment

```
nvcc buffon.cu -o buffon -O2 -arch=native
./buffon -n 1e9                 # GPU, all SMs, double precision
./buffon -n 1e9 -m 4 -p 32      # 4 SMs, float precision
./buffon -n 1e8 -c              # one CPU core (EP1)
./buffon -i                     # GPU properties
```

Prints one line of `KEY=VALUE` (π, real error, SIGMA, times, cycles).

## Experiments

```
python buffon.py                     # GPU, FP64, 10 minutes
python buffon.py --precision 32      # or 16
python buffon.py --cpu               # EP1 on the CPU
python buffon.py --budget 30m        # longer
python buffon.py --report runs/A runs/B ...
```

Each run sweeps N from 10⁴ while the budget allows, with several seeds per N and, on the GPU, every SM count (1, 2, 4, … all).

## Output

```
runs/<date>_<time>-<device>-fp<precision>/
    config.json    settings and GPU
    results.csv    one line per engine run
    error.png      real vs experimental error; run time vs N
    amdahl.png     speedup vs SMs; p vs N (GPU)
    prediction.png time for each decimal of π, with the prediction test
    summary.txt    fits, Amdahl, time for 14 decimals
runs/<date>_<time>-report/
    report.png     error of all runs; time for each decimal of π
    report.txt     speeds and times, side by side
```

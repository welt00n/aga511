# EP4: the spotted star, toward the GPU

The EP2 star (Monte Carlo image of a star with spots, the professor's algorithm) rebuilt with CUDA: the same engine on the CPU and on the GPU, checked against each other bit for bit, measured over the GPU's SMs (Amdahl and Gustafson), and a 1k showcase. Results below; the plan and its log in [`../docs/roadmap.md`](../docs/roadmap.md).

The ideas behind every choice are in [`../docs/README.md`](../docs/README.md); the GPU design in [`../docs/gpu_engine.md`](../docs/gpu_engine.md) and [`../docs/hpc_architecture.md`](../docs/hpc_architecture.md).

## Requirements

- the CUDA toolkit (`nvcc`) and an NVIDIA GPU (without one, `test` runs the CPU half);
- the conda env `aga511`, from conda-forge only: numpy, matplotlib, pillow, pandas, ipykernel, nbformat, nbclient, nbconvert; tectonic and pandoc for the PDF; ffmpeg for the videos.

## How to run: one entrypoint, four products

```
python main.py                                   a menu: it asks, then prints the equivalent command
python main.py test                              does it work here? (~1 min)
python main.py make --snr 10                     pictures and videos of the star (1024², 300 frames by default)
python main.py measure                           every measurement on this machine, with its report (~70 min on the laptop)
python main.py measure --parts kernel,occupancy  only some parts
python main.py compare runs/measure/A runs/measure/B    two or more machines side by side
python main.py runs                              every run so far, newest first
python main.py build --arch 80                   build every engine (here: for the A30)
```

| Product | Question | What it leaves in `runs/<product>/<stamp>-<label>/` |
|---|---|---|
| `test` | does it work here? the environment, the builds, the random numbers, bit for bit (split, passes, GPU = CPU), and the GPU vs the sequential CPU with independent photons (χ²) | `summary.md`: pass/fail and what each test proves |
| `make` | the pictures (by default 7 observers, from the north pole to the south pole every 30°) | an mp4 and a still per observer; all observers side by side (`observers_grid.mp4`) and placed around the star at their inclination (`observers_around.mp4`); each video repeats the rotation `--loops` times (5 by default); the SNR measured vs predicted |
| `measure` | how fast, how it scales, and why, at valid N | `report.pdf` and `report.ipynb`, `points/`, `parts/`, `figures/` |
| `compare` | how two machines differ | `report.pdf`, `report.ipynb`, the comparison's data and figures |

- **Every product checks first and predicts:** the environment (the compiler, the GPU, other GPU work, disk, the packages, uncommitted engine code, whether the speeds were measured here), the GPU and the launch the engine will use (SMs, threads and warps per SM, registers; per precision: registers per thread, threads per block × blocks per SM, warps per SM, occupancy, threads in all), then the time, the memory, the passes and the disk of what it will run. Then it asks. `--plan` stops after the prediction; `--yes` doesn't ask.
- **A valid measurement has a minimum N per device:** at least 1 s of photons, and at least 32 chunks of 1024 photons per resident GPU thread. On the laptop, on all SMs: ~1·10⁹ (FP64), ~1·10¹⁰ (FP32). `measure` refuses to run below it, and every timed run in it is above it; `--allow-invalid` makes a report labelled "not valid", for trying things out. Why: [`../docs/research.md`](../docs/research.md) (Q3).
- **Image sizes: 1k minimum for measurements, 2k maximum.** `measure` traces, records and saves the showcase's own image (1024² × 300 frames × 4 observers, 5 GB per run; `--size` 1024–2048), so its serial part is that of a real high-resolution picture. `make` goes up to 2048² (`--size`); 2k is the goal, run on the A30 (on the laptop, 7 observers at 2k need 7 passes, ~3 h).
- **The unit of speed is the SM:** the GPU is measured against itself (one SM, then all of them), never against the CPU. The sequential CPU engine is there for correctness only: the GPU's images must equal its images bit for bit, and agree with independent ones statistically. Comparing EP4 with EP2 is a separate question, for the TCD.
- **Each run is a folder** with `run.json` (what was asked, the machine, the status) and `summary.md` (its page); `runs/index.md` lists them all. A measurement shares its runs between parts: the reference run on all SMs is the speed row, the kernel's baseline, the uncapped register run and the strong sweep's 36-SM point, run once. `measure --continue <folder>` runs only what is missing.
- **The engines build themselves** when the code changed: one `nvcc` call per binary, no Makefile; the code hash covers the engine's sources, the command and the `nvcc` version. Next to each binary in `engine/build/`: `.hash` and `.report` (the compiler's resource report).
- **Standalone research scripts,** one per question we investigated, are in [`research/`](research/); their log is [`../docs/research.md`](../docs/research.md).

## Engine: one run

```
B=engine/build
$B/star_cpu64 --test-rng         # Philox against the Random123 test vectors
$B/star_cpu64 -n 1e8 -o out      # 1e8 photons, EP2's star, 4 observers, one file per observer in out/
$B/star_cpu32 -n 1e8 -o out32    # the same photons in FP32
$B/star_cpu64 -n 1e8 --observers 1.5708,0.7854,0 --size 512 --frames 300   # inclinations in radians
$B/star_cpu64 -n 1e8 -o out --observers-per-pass 1      # one observer per pass: 1/4 of the memory
$B/star_cpu64 -n 1e8 -o out --memory-budget 2GB         # or as many observers per pass as fit (also 1.5GiB)
$B/star_cpu64 -n 1e8 --memory-budget 50MB --describe    # what would run: nmu, bins, passes, bytes; runs nothing
$B/star_cpu64 -n 1e8 --first-chunk 0 --last-chunk 48828 # part of the run
$B/star_cpu64 -n 1e8 --no-record # the same photons, without an image
$B/star_gpu64 -n 1e8 --describe  # the GPU: PROCESSOR=GPU, its name, compute capability, SMs
$B/star_cpu64 -h                 # all options
```

- **`--describe`** prints a `CONFIG` line (the derived configuration, `OBSERVER_BYTES`, `IMAGE_BYTES`) and one `OBSERVER` line per observer (μ bin, bin center, its pass, its file name), then exits.
- **Passes are balanced:** with 4 observers and `--observers-per-pass 3` the passes are 2 + 2, not 3 + 1 (the same time, less memory).
- **Output lines,** `KEY=VALUE` after a first word that says what the line is:
  - `PROGRESS` at most every 10 s: pass, chunk, speed, elapsed, ETA;
  - `PASS` after each pass: its observers' μ bins, photon counts, `ZERO_TIME`, `PHOTONS_TIME`, `COPY_TIME`, `SAVE_TIME`;
  - `SUMMARY` last: the processor (`PROCESSOR`, `PROCESSOR_NAME`, `COMPUTE_CAPABILITY`, `COMPUTE_UNITS`, `UNIT`), the configuration, the counts (`PHOTONS`, `BLOCKED` by spots, `MISSED` the observers, `RECORDED`), the phase times (`SETUP_TIME`, `ZERO_TIME`, `PHOTONS_TIME`, `COPY_TIME` to the host, `SAVE_TIME`, `TIME`) and `PHOTONS_PER_SECOND`.
- **Per observer:** `observer_mu0023_i1.5708.bin` (raw `uint32` counts, `[frame][iy][ix]`) and `observer_mu0023_i1.5708.record` (`KEY=VALUE`, one per line, written last: the "done" marker). The name carries the μ bin and the inclination of the bin center.
- **Each file is written as `.partial`, put on the disk (`fsync`), then renamed:** a file that exists is whole, even after a power cut.
- **SETUP touches every page of the image;** each pass's `ZERO_TIME` is then pure zeroing, the same in every pass.
- **Passes** re-simulate the same photons, so any number of passes gives the same files, bit for bit. Design: [`../docs/jobs_and_passes.md`](../docs/jobs_and_passes.md).

## How the code is organized

**The engine** (`engine/`, C++/CUDA; includes carry their folder, `nvcc` runs in `engine/` with `-I.`):

| File | What it holds |
|---|---|
| `main.cu` | the pass loop and its phase timings |
| `processor.cuh` | what the pass loop asks of a processor (open, load a pass, zero, trace, hand an observer to the host, close) |
| `physics/real.cuh` | the `real` type (FP64, or FP32 with `-DSTAR_FP32`), math functions for both, the `HD` qualifier (`__host__ __device__` under nvcc) |
| `physics/geometry.cuh` | 3D vectors |
| `physics/photon.cuh` | the physics of one photon, from random numbers to a pixel index. Pure: it records nothing |
| `physics/scene.cuh`, `scene.cu` | the `Scene`: flat and read-only (spots, limb darkening, which μ bins this pass records), ready for `__constant__` memory |
| `physics/observers.cuh`, `observers.cu` | observers: inclination → μ bin, bin center, file name; passes |
| `random/rng.cuh` | a copy of cuRAND's Philox4_32_10 for the CPU: same init, counter layout and conversions, so the CPU draws the numbers cuRAND draws on the GPU |
| `random/gpu_rng.cuh` | cuRAND's Philox as the physics' generator on the GPU |
| `random/rng_test.cuh`, `rng_test.cu`, `gpu_rng_test.cu` | `--test-rng`: Random123's known answers, and on the GPU cuRAND vs our copy, bit for bit |
| `run/options.cuh`, `options.cu` | the command line, observers per pass from a memory budget |
| `run/run.cuh` | the types of a run's results (counts, times, progress, the processor's description) and the clock |
| `run/image.cuh`, `image.cu` | the image in host memory: allocate, zero, `count_photon`; saving a block of counts |
| `run/files.cuh` | write to `.partial`, `fsync`, then rename: a file that exists is always whole |
| `run/report.cuh`, `report.cu` | the output lines and the observer records |
| `cpu/cpu.cu` | the CPU as a processor: the chunk loop with our Philox copy |
| `gpu/gpu.cu` | the GPU as a processor: the Scene in constant memory, the kernel `trace_photons` (EP3's persistent threads and queue around the shared physics), the launch, the pinned host buffer |
| `gpu/gpu_image.cuh`, `gpu_image.cu` | the image in device memory: allocate, zero, copy one observer back; `count_photon_atomic` |
| `gpu/gpu_sms.cuh`, `gpu_sms.cu` | the SMs: `%smid`, the mask (`--sms N`), the work per SM (`SM`, `SMS` lines) |
| `gpu/gpu_device.cuh`, `gpu_device.cu` | the GPU itself: CUDA errors, properties, name, compute capability, SMs |
| `gpu/gpu_sections.cuh` | the section timers (laps with `clock64()`), only in the `sections` build variant |

**The products** (`main.py` and `lab/`, Python; `argparse` only in `main.py`):

| File | What it holds |
|---|---|
| `main.py` | the only entrypoint: the commands, their options, the menu |
| `lab/testing.py`, `make.py`, `measurement.py`, `comparison.py` | the four products |
| `lab/environment.py` | the checks before anything runs |
| `lab/plan.py` | a run's prediction (time, memory, passes, disk), the minimum N and why, the confirmation |
| `lab/points.py` | running one engine process and keeping its result (`points/<run>.json`), skipping what is done |
| `lab/parts.py` | what each part of a measurement runs, and its analysis (`parts/<part>.json`, `figures/`) |
| `lab/explainer.py`, `report.py` | the measurement's report: its sections (question, method, raw table, figure, meaning, caveats), the notebook, the PDF |
| `lab/engine.py` | building the engines and talking to them: run, `--describe`, records |
| `lab/runs.py`, `speeds.py` | the run folders and the index; the speeds measured on each host (for the predictions) |
| `lab/checks.py`, `analysis.py` | the correctness checks (split, passes, precision, GPU = CPU bit for bit, GPU vs CPU with independent photons); images and comparisons |
| `lab/scaling.py` | Amdahl and Gustafson: the fits and figures |
| `lab/device.py` | the GPU's limits, registers → warps → occupancy, compile-only targets (the A30) |
| `lab/instruction_mix.py` | the kernel's machine instructions counted by hardware unit |
| `lab/gpu_monitor.py` | `nvidia-smi` sampled during a run: clocks, power, temperature, the clock-limit reasons |
| `lab/videos.py` | the videos: one per observer, the grid, the views around the star; one rotation encoded once, then repeated |
| `lab/snr.py`, `star.py` | N from the SNR, the SNR measured vs predicted; describing a star and its phases |
| `research/` | one standalone script per investigation, and their shared `common.py` ([`../docs/research.md`](../docs/research.md)) |

## What changed from EP2, and why

- **Random numbers.** Philox (cuRAND's generator) replaces `rand()`. One stream per **chunk** of photons: chunk `c` uses subsequence `c`, so a photon's numbers never depend on who simulates it or in which order.
- **Chunks.** A fixed size (1024 photons, `--chunk-size`), and `--first-chunk`/`--last-chunk` to run part of the work. Partial images add up exactly to the full one.
- **The physics is pure.** `trace_photon` returns a pixel index and writes nothing, so the GPU can change how photons are recorded without touching the physics.
- **The Scene is computed once.**
  - Spot thresholds are computed once, not per photon (EP2 called `cos()` per photon per spot).
  - The observer table replaces the search loop.
  - Spots come from a reserved random stream, so every chunk range sees the same star.
- **FP32-ready.**
  - `real` switches between double and float.
  - `sqrt((1 - x)(1 + x))` replaces `sqrt(1 - x*x)`.
  - Spots are tested by squared chord, not by cosines close to 1.
- **The observer is checked before the pixel.** ~97% of photons fly toward no observer and skip the pixel computation.
- **Phase timings** (setup / zero / photons / save): the serial and parallel parts of Amdahl's law. The image is zeroed with `memset` at the start of each pass, so its cost is measured (`ZERO_TIME`).
- **Observers by inclination, in radians,** not by count. `nmu` defaults to the odd number nearest frames/π, so the equator-on observer sits at a bin center.
- **Passes and one file per observer,** so an image bigger than the memory can be done in pieces.
- **Fixes.**
  - The photon count is `long long` (EP2's `Image.N` was `unsigned int`).
  - Two observers in the same mu bin are an error (EP2 silently gave one of them nothing).

## First results (CPU, Intel Core Ultra 7 255HX, one core)

| Check | Result |
|---|---|
| Philox, Random123 test vectors | 3/3 pass |
| Two halves of the chunk range vs the full run (N = 10⁷) | identical, bit for bit |
| EP4 vs EP2 without spots (N = 2·10⁸, independent random numbers) | χ²/dof = 1.0047 over 126 008 pixels (+1.2σ from 1): the same physics |
| Speed, FP64 | 1.0·10⁷ photons/s (EP2: 7.6·10⁶, including its setup and save) |
| FP32 vs FP64, same photons (N = 10⁸) | 5·10⁻⁴ of the recorded photons land in a different pixel, against 0.22 Poisson noise per lit pixel; FP32 is 1.58× faster on the CPU |

### With passes (2026-11-21, EP2's star, 4 observers at 1.5708, 1.0472, 0.5236, 0 rad, nmu = 47)

| Check | Result |
|---|---|
| 4 observers in 1 pass vs 4 passes of 1 (N = 10⁷) | the same counts, identical images bit for bit |
| Two halves of the chunk range vs the full run (N = 10⁷) | identical, bit for bit |
| EP4 vs EP2 without spots (N = 2·10⁸, EP2's observers) | χ²/dof = 0.9957 over 126 496 pixels (−1.1σ from 1) |
| Speed, FP64 | 9.1·10⁶ photons/s recording 4 observers, 1.03·10⁷ recording one: writing to the image costs ~180 ns per recorded photon, and one observer writes 1/4 as many ([details](../docs/jobs_and_passes.md#what-we-learned)) |
| FP32 vs FP64 (N = 10⁸) | 1.2·10⁻⁴ of the photons move; FP32 1.47× faster. Less than before because the pole-on observer is farther from the pole ([details](../docs/jobs_and_passes.md#15-implemented-the-engine-with-passes-cpu-2026-11-21)) |

At N = 10⁸, one frame has ~0.14 photons per pixel (see the planning formula in the docs): the single-frame pictures are still noise. The GPU is what will make clean frames possible.

## Results on the GPU (2026-11-21, RTX 5070 Laptop GPU, 36 SMs)

**Correctness**

| Check | Result |
|---|---|
| cuRAND on the GPU vs our CPU copy | the same numbers, bit for bit, raw and uniform, FP64 and FP32, chunks up to 2³² + 5 |
| GPU vs CPU, the same photons | **FP64: identical images, bit for bit**; FP32: 3.2·10⁻⁵ of the photons in another pixel (3/4 from fused multiply-add) |
| split, passes, on the GPU | bit for bit |
| the same image on 1, 2, 9, 36 SMs | byte for byte |
| EP4 (GPU) vs EP2 | χ²/dof = 0.9957, the CPU's number |

**Speed and scaling:** `python main.py measure`, 2026-11-22, on the showcase's image (**1024² × 300 frames × 4 observers, 5 GB per run**), all parts at valid N, 3 repeats, medians; 227 engine runs in 89 min (predicted 88). The report: [`runs/measure/2026-11-22_20-25-24-Welton-all/report.pdf`](runs/measure/2026-11-22_20-25-24-Welton-all/report.pdf). In brackets: the earlier measurement on a small image (200² × 150 frames, 0.1 GB per run, `runs/measure/2026-11-22_17-08-40-Welton-all/`).

| | FP64 (N = 4·10⁹) | FP32 (N = 5·10¹⁰) |
|---|---|---|
| photons/s, one SM | 1.84·10⁷ | 3.15·10⁸ |
| photons/s, all 36 SMs | 6.69·10⁸ | 1.11·10¹⁰ (the power cap holds the clock at ~2020 MHz) |
| all 36 SMs vs one SM, photon phase | **36.3×** [35.2×]; 36.1× in cycles | **35.3×** [32.9×]; 36.3× in cycles |
| FP32 / FP64 speed | 17.1× on one SM, 16.6× on all SMs | |
| strong scaling, **whole run** (the photons + zeroing, copying back and saving 5 GB) | **19.3×** [33.7×]; Amdahl p = 0.974, ceiling 38; serial share measured 0.026 | **16.3×** [31.4×]; p = 0.971, ceiling 35; serial 0.034 |
| weak scaling, 1 → 36 SMs | efficiency 0.993; Gustafson s′ 0.48 (measured 0.48) [0.055] | efficiency 0.939; s′ 0.57 (measured 0.55) |
| p(N), photon phase (FP32, 1 vs 36 SMs) | | −0.002 (10⁵), 0.14 (10⁶), 0.65 (10⁷), 0.97 (10⁸), 0.9985 (10¹⁰) |

- **The image size moved Amdahl, not the photons.** The photon phase still scales almost perfectly (36×), but on all SMs a 1k run is 6.0 s of photons and 5.6 s of zeroing, copying back and saving the 5 GB image. That serial part doesn't get faster with SMs, so the whole run gains only 19× (FP64). On the small image it was 34×. For high-resolution pictures, the disk is the serial part.
- **Weak scaling shows it too:** with N growing with the SMs, the image doesn't grow, so on 36 SMs the serial share is ~48% of the run (Gustafson s′ 0.48, measured and fitted agree).

**Correctness at valid N (1k image):** the GPU vs the sequential CPU with independent photons (the same star, different chunks; compared in 8×8-pixel blocks so that each holds enough photons), χ²/dof 0.990; GPU = CPU bit for bit in FP64; FP32 moves 7.2·10⁻⁴ of the photons (smaller pixels, so more photons cross an edge: 1.1·10⁻⁴ at 200²), below the noise per lit pixel (0.14); the GPU's FP32 vs the CPU's 3.0·10⁻⁴.

**The kernel and occupancy:** 80 / 67 registers, no spills, occupancy 0.50 / 0.58 (the registers limit it). The scene in global memory: no difference. No FMA: 1.09× the time in FP64, the same image. Capping the registers: FP64 is the same speed down to 64 registers (occupancy 0.67; 72 registers is 0.8% fewer cycles), then the spills cost up to 9% at full occupancy; FP32 is fastest uncapped. On this GPU, more warps don't help: FP64 waits on its few FP64 units, not on latency. The A30 (compiled here): 96 / 76 registers, occupancy at most 0.33 / 0.39 ([`../docs/research.md`](../docs/research.md)).

**The showcase** (`python main.py make --size 1024 --frames 300 --snr 10`, 2026-11-22): 1024², 300 frames, 4 observers in one pass (5.03 GB on the GPU), **N = 3.9·10¹² photons in 378 s** (predicted 6 min; 385 s on 2026-11-21); SNR measured = predicted (14.0 / 14.1 at the center, 11.7 / 11.7 near the limb); four 10 s videos of ~46 MB (kept out of git) and their stills in [`runs/make/2026-11-22_18-17-47-1024-f300-gpu32-snr10/`](runs/make/2026-11-22_18-17-47-1024-f300-gpu32-snr10/).

The full story, piece by piece: [`../docs/walkthrough.md`](../docs/walkthrough.md), pieces 16–25.

## Next

The driver in full (jobs, resuming, remote runs at IAG: [`../docs/driver.md`](../docs/driver.md)), the A30, the CPU on all its cores ([`../docs/roadmap.md`](../docs/roadmap.md)).

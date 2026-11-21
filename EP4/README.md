# EP4: the spotted star, toward the GPU

The EP2 star (Monte Carlo image of a star with spots, the professor's algorithm) rebuilt with CUDA. **In progress (2026-11-21):** the CPU engine is complete; the GPU engine is being built ([`../docs/roadmap.md`](../docs/roadmap.md) §0).

The ideas behind every choice are in [`../docs/README.md`](../docs/README.md); the GPU design in [`../docs/gpu_engine.md`](../docs/gpu_engine.md) and [`../docs/hpc_architecture.md`](../docs/hpc_architecture.md).

## Requirements

- CUDA toolkit (`nvcc`)
- Python 3 with numpy, matplotlib, pillow (conda env `aga511`)

## Building

```
python engine.py              # builds star_cpu64, star_cpu32, star_gpu64, star_gpu32 in engine/build/, printing each nvcc command
python engine.py --arch 80    # for the A30 (SASS for compute capability 8.0 only)
```

- **No Makefile:** `engine.py` composes one `nvcc` call per binary and rebuilds only when the code hash (the engine's sources, the command, the `nvcc` version) changes. `star.py` builds what it needs by itself.
- **`star_cpu*` link `cpu.cu`, `star_gpu*` link `gpu.cu`:** the same pass loop, two processors ([`../docs/gpu_engine.md`](../docs/gpu_engine.md)).
- Next to each binary: `.hash` (what built it) and `.report` (the compiler's resource report for the GPU kernels).

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

## Driver

```
python star.py -n 1e8                       run, a gif and a still frame per observer
python star.py -n 1e8 --precision 32        the FP32 build
python star.py -n 1e8 --processor gpu       on the GPU (every check takes --processor gpu too)
python star.py -n 1e8 --compare-gpu         the GPU vs the CPU on the same photons
python star.py -n 2e8 --compare-ep2         same physics as EP2? (chi2, no spots)
python star.py -n 1e7 --check-split         two halves of the chunk range = the full run?
python star.py -n 1e7 --check-passes        one observer per pass = all in one pass?
python star.py -n 1e8 --compare-precision   FP32 vs FP64 on the same photons
```

This is still the first, simple driver. The full one (jobs, resuming, remote runs, sweeps) is designed in [`../docs/driver.md`](../docs/driver.md).

Each run gets a folder in `runs/`: `run.json` and the gifs for a simulation, `report.txt` (and `comparison.png`) for a check.

## How the code is organized

| File | What it holds |
|---|---|
| `engine/real.cuh` | the `real` type (FP64, or FP32 with `-DSTAR_FP32`), math functions for both, the `HD` qualifier (`__host__ __device__` under nvcc) |
| `engine/rng.cuh` | a copy of cuRAND's Philox4_32_10 for the CPU: same init, counter layout and conversions, so the CPU draws the numbers cuRAND draws on the GPU |
| `engine/geometry.cuh` | 3D vectors |
| `engine/scene.cuh`, `scene.cu` | the `Scene`: flat and read-only (spots, limb darkening, which μ bins this pass records), ready for `__constant__` memory |
| `engine/observers.cuh`, `observers.cu` | observers: inclination → μ bin, bin center, file name; passes |
| `engine/photon.cuh` | the physics of one photon, from random numbers to a pixel index. Pure: it records nothing |
| `engine/image.cuh`, `image.cu` | the image in host memory: allocate, zero, `count_photon`; saving a block of counts |
| `engine/files.cuh` | write to `.partial`, `fsync`, then rename: a file that exists is always whole |
| `engine/options.cuh`, `options.cu` | the command line, observers per pass from a memory budget |
| `engine/run.cuh` | the types of a run's results (counts, times, progress, the processor's description) and the clock |
| `engine/report.cuh`, `report.cu` | the output lines and the observer records |
| `engine/processor.cuh` | what the pass loop asks of a processor (open, load a pass, zero, trace, hand an observer to the host, close) |
| `engine/cpu.cu` | the CPU as a processor: the chunk loop with our Philox copy |
| `engine/gpu.cu` | the GPU as a processor: the Scene in constant memory, the kernel `trace_photons` (EP3's persistent threads and queue around the shared physics), the launch, the pinned host buffer |
| `engine/gpu_image.cuh`, `gpu_image.cu` | the image in device memory: allocate, zero, copy one observer back; `count_photon_atomic` |
| `engine/gpu_rng_test.cu` | `--test-rng` on the GPU: cuRAND vs `rng.cuh`, bit for bit |
| `engine/gpu_sms.cuh`, `gpu_sms.cu` | the SMs: `%smid`, the mask (`--sms N`), the work per SM (`SM`, `SMS` lines) |
| `engine/gpu_device.cuh`, `gpu_device.cu` | the GPU itself: CUDA errors, properties, name, compute capability, SMs |
| `engine/main.cu` | the pass loop and its phase timings |
| `engine/rng_test.cuh`, `rng_test.cu` | `--test-rng`: Random123's known answers, and on the GPU cuRAND vs our copy |
| `engine/gpu_rng.cuh` | cuRAND's Philox as the physics' generator on the GPU |
| `engine.py` | building the engines (no Makefile) and talking to them: run, `--describe`, records |
| `star.py`, `analysis.py` | the command line: runs with pictures, the checks; images and comparisons |
| `engine/gpu_sections.cuh` | the section timers (laps with `clock64()`), only in the `sections` build variant |
| `gpu_monitor.py` | `nvidia-smi` sampled during a run: clocks, power, temperature, utilization, memory |
| `instruction_mix.py` | the GPU kernel's compiled instructions counted by hardware unit (`python instruction_mix.py`) |
| `kernel_experiments.py` | the kernel's variants (Scene in global memory, capped registers, no FMA, section timers) timed and compared (`python kernel_experiments.py`) |

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

## Next

The driver, local then remote ([`../docs/driver.md`](../docs/driver.md), steps 3–4). Then the GPU step: kernels for the chunk loop, cuRAND on the device (checked against this CPU copy bit for bit), the queue and the image in device memory, the Scene in constant memory, and the SM sweep for Amdahl's law.

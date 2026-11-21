# EP4: the spotted star, toward the GPU

The EP2 star (Monte Carlo image of a star with spots, the professor's algorithm) rebuilt so it can move to CUDA. **This first step is sequential**: no CUDA code yet. It is the EP2 physics written in the shape the GPU needs, compiled with `nvcc`.

The ideas behind every choice are in [`../docs/README.md`](../docs/README.md) and [`../docs/precision.md`](../docs/precision.md).

## Requirements

- CUDA toolkit (`nvcc`), used here as the compiler
- Python 3 with numpy, matplotlib, pillow (conda env `aga511`)

## Engine: one run

```
cd engine
make                      # star (FP64) and star32 (FP32)
./star --test-rng         # Philox against the Random123 test vectors
./star -n 1e8             # 1e8 photons, EP2's default star, writes star.bin
./star32 -n 1e8           # the same photons in FP32
./star -n 1e8 --first-chunk 0 --last-chunk 48828   # part of the run
./star -n 1e8 --no-record # the same photons, without writing the image
./star -h                 # all options
```

Prints one line of `KEY=VALUE` pairs: the configuration, the photon counts (`PHOTONS`, `BLOCKED` by spots, `MISSED` the observers, `RECORDED`), the phase times (`SETUP_TIME`, `PHOTONS_TIME`, `SAVE_TIME`, `TIME`) and `PHOTONS_PER_SECOND`. The image is raw `uint32` counts, laid out `[iphi][observer][iy][ix]` as in EP2.

## Driver

```
python star.py -n 1e8                       run, a gif and a still frame per observer
python star.py -n 1e8 --precision 32        the FP32 build
python star.py -n 2e8 --compare-ep2         same physics as EP2? (chi2, no spots)
python star.py -n 1e7 --check-split         two halves of the chunk range = the full run?
python star.py -n 1e8 --compare-precision   FP32 vs FP64 on the same photons
```

Each run gets a folder in `runs/`: `run.json` and the gifs for a simulation, `report.txt` (and `comparison.png`) for a check.

## How the code is organized

| File | What it holds |
|---|---|
| `engine/real.cuh` | the `real` type (FP64, or FP32 with `-DSTAR_FP32`), math functions for both, the `HD` qualifier (`__host__ __device__` under nvcc) |
| `engine/rng.cuh` | a copy of cuRAND's Philox4_32_10 for the CPU: same init, counter layout and conversions, so the CPU draws the numbers cuRAND draws on the GPU |
| `engine/geometry.cuh` | 3D vectors |
| `engine/scene.cuh`, `scene.cu` | the `Scene`: flat and read-only (spots, limb darkening, observer table), ready for `__constant__` memory |
| `engine/photon.cuh` | the physics of one photon, from random numbers to a pixel index. Pure: it records nothing |
| `engine/image.cuh`, `image.cu` | the image: allocate, `count_photon` (`counts[i]++`, later `atomicAdd`), save |
| `engine/options.cuh`, `options.cu` | the command line |
| `engine/main.cu` | the chunk loop, phase timings, the result line, the RNG test |
| `star.py`, `analysis.py` | driver: build, run, pictures, checks |

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
- **Phase timings** (setup / photons / save): the serial and parallel parts of Amdahl's law. The image is zeroed with `memset`, not `calloc`, so its cost is counted in setup (see `image.cu`).
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

At N = 10⁸, one frame has ~0.14 photons per pixel (see the planning formula in the docs): the single-frame pictures are still noise. The GPU is what will make clean frames possible.

## Next

The GPU step: kernels for the chunk loop, cuRAND on the device (checked against this CPU copy bit for bit), the queue and the image in device memory, the Scene in constant memory, and the SM sweep for Amdahl's law.

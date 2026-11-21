# EP4 walkthrough

A guided tour of the EP4 sequential engine, one piece at a time, with the questions that came up on the way. Code: [`../EP4/engine/`](../EP4/engine/).

| # | Piece | File(s) | Status |
|---|---|---|---|
| 1 | One run from start to finish | `main.cu` | done |
| 2 | The `real` type and `HD` | `real.cuh` | done |
| 3 | Random numbers: Philox | `rng.cuh` | next |
| 4 | Chunks: dividing the work | `main.cu` | |
| 5 | The Scene: the star, prepared once | `scene.cuh`, `scene.cu` | |
| 6 | One photon: the physics | `photon.cuh` | |
| 7 | The image and the timings | `image.cu`, `main.cu` | |
| 8 | The command line | `options.cu` | |
| 9 | The driver and the three checks | `star.py`, `analysis.py` | |
| 10 | Docs and the class plan | `docs/`, `cuda_class/` | |

---

## Piece 1: one run from start to finish

`./star -n 1e8` does four things, in order, inside `main`:

```
load_options          read the command line            → Options
─── setup ─────────── (timed: SETUP_TIME)
build_scene           place spots, fill observer table → Scene
allocate_image        malloc + memset the counts       → Image
─── photons ───────── (timed: PHOTONS_TIME)
simulate_chunks       for every chunk: for every photon:
                          trace_photon → pixel, or "blocked", or "missed"
                          count_photon(pixel)
─── save ──────────── (timed: SAVE_TIME)
save_image            write the counts to star.bin
print_result          one KEY=VALUE line
```

- **The three timed phases are Amdahl's law in advance.**
  - Setup and save are the serial part *s*; the photons are the part *p* the GPU will run in parallel.
  - In CUDA only the middle block changes. Setup will also copy the Scene to the GPU, and save will also copy the image back: more serial time, already measured separately.
- **`main` only connects the pieces.** Each step is a function named for what it does, so `main` reads like the list above. You open a function only when you want its details.

What a run prints at the end (N = 10⁸, EP2's default star):

```
PHOTONS=100000000 BLOCKED=411940 MISSED=96936043 RECORDED=2652017
SETUP_TIME=0.039 PHOTONS_TIME=9.8 SAVE_TIME=0.042 PHOTONS_PER_SECOND=1.02e+07
```

- **`BLOCKED`:** emitted inside a spot (0.4%).
- **`MISSED`:** flew toward a μ bin with no observer (97%).
- **`RECORDED`:** landed in an observer's image (4 observers / 150 μ bins = 2.65%).
- **On the CPU**, setup + save ≈ 0.08 s against ~9.8 s of photons, so p ≈ 0.99. On the GPU the photons shrink a lot while setup and save stay about the same, so *s* matters much more. That is the story of the TCD.

### Questions on piece 1

**How are these numbers computed? Is it expensive?**

- **The photon counts.** `trace_photon` already returns one of three results, and `simulate_chunk` adds 1 to the matching counter.
  - That is one integer increment per photon, kept in a register, against ~50 floating-point operations (4 sin/cos, a sqrt, a cbrt, an atan2).
  - It costs well under 1%.
- **The times.** `CLOCK_MONOTONIC` is read 6 times per run, around the phases, never per photon: ~20–50 ns each.
- **The output.** One `printf` at the very end, after all timing has stopped.
- **Rule:** count in the hot loop; time and print only outside it.
  - Logging is expensive when it is *inside* the loop: 10⁸ prints would take longer than the physics, and 10⁸ clock reads at ~30 ns would add ~3 s.
  - On the GPU it is worse: `printf` in a kernel serializes threads, and timing small pieces with `cudaDeviceSynchronize()` stops the whole GPU each time.
- **On the GPU, the counters need care.**
  - One `atomicAdd` per photon on three global counters would be 10¹² atomics on the same three addresses: the counters would become the bottleneck.
  - Instead, as EP3 does with `crossings`, each thread counts in registers and does one `atomicAdd` per counter at the end: ~10⁵ atomics instead of 10¹².
  - That is "cache in the thread, write once at the end", and here it works because every photon adds to the same few counters.

**Are we using Nsight already?**

Not for EP4 yet: Nsight is mainly for GPU code, and there is no kernel yet. For the CPU code our own phase timers are enough. EP3's Buffon has been profiled with `nsys`: see [`../cuda_class/nsight.md`](../cuda_class/nsight.md) for how the tools work and what we found (the serial part of Buffon is CUDA context creation).

---

## Piece 2: `real.cuh`, the precision switch

The smallest file, and every other file depends on it. Three things:

**1. One type for the physics.**

```c
#ifdef STAR_FP32
typedef float real;      // make star32  (-DSTAR_FP32)
#else
typedef double real;     // make star
#endif
```

All the physics uses `real`, never `double` or `float` directly. The Makefile builds the same source twice. One copy of the physics, two precisions: the FP32 vs FP64 comparison compares exactly the same code.

**2. One name per math function.**

```c
static inline HD float  real_sqrt(float x)  { return sqrtf(x); }
static inline HD double real_sqrt(double x) { return sqrt(x); }
```

C++ overloading picks the version from the argument's type.

- **Without it:** in C, `sqrt(x)` on a float converts to double, computes in FP64 and converts back.
- **On the CPU** that is a small slowdown.
- **On a GPU with slow FP64 it is a trap:** an "FP32" kernel would secretly run FP64 math, at 1/64 of the speed on the laptop.
- Going through `real_*` makes that mistake impossible.

**3. `HD`: one function for CPU and GPU.**

```c
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif
```

- `__host__ __device__` tells `nvcc` to compile the function for **both** processors.
- Every physics function is marked `HD`, so the future kernel calls the same `trace_photon` the CPU calls today, with no second copy to keep in sync.
- Outside `nvcc` the macro is empty, and the code still compiles as plain C++.

**Plus one helper: `sine_from_cosine`.**

```c
static inline HD real sine_from_cosine(real cosine) {
    real sine_squared = (1 - cosine) * (1 + cosine);
    return sine_squared > 0 ? real_sqrt(sine_squared) : 0;
}
```

The FP32-safe version of `sqrt(1 - c²)`, used three times: the ring radius from z, sin θ from μ, and the professor's `raiz` from w.

### Questions on piece 2

**Why `(1 - c)(1 + c)` and not `1 - c*c`? Does it matter in FP32?**

Full answer, with measurements: [precision.md, section 8](precision.md#8-deep-dive-sine_from_cosine-and-the-fp32-correction). In short:

- **The naive form suffers catastrophic cancellation.** The rounding of `c*c` survives the subtraction, and the result 1 − c² ≈ θ² is small. The relative error grows like u/(2θ²).
- **The factored form keeps a few units of roundoff at every angle.** `1 - c` is exact (Sterbenz's lemma).
- **Measured in FP32:** the worst case is 4.3·10⁻⁵ (naive) vs 1.0·10⁻⁷ (factored). Surprisingly, the naive worst case is near θ ≈ 1°, not at the pole.
- **Neither form can recover what the cosine lost.** Below θ ≈ 2.4·10⁻⁴ rad, cos θ rounds to 1 in FP32.
- **In the star, the two forms give practically the same images** (5.01 vs 5.04·10⁻⁴ of photons moved).
  - The real FP32 sensitivity is in the pole observers (~30× more photons change pixel).
  - The cause is the projection, which divides by `raiz` near a coordinate singularity.
- **Lessons:**
  - rewrite cancellations when it is free;
  - precision belongs to the whole calculation;
  - measure before fixing.

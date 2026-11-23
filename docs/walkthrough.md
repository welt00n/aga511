# EP4 walkthrough

A guided tour of the EP4 sequential engine, one piece at a time, with the questions that came up on the way. Code: [`../EP4/engine/`](../EP4/engine/).

- **Pieces 1–9** describe the engine of phase B (commit `d945e97`).
- **Pieces 10–15** describe what phase C added or changed: passes, observers in radians, one file per observer ([roadmap.md](roadmap.md)). Where they change pieces 7–9, they say so.

| # | Piece | File(s) | Status |
|---|---|---|---|
| 1 | One run from start to finish | `main.cu` | done |
| 2 | The `real` type and `HD` | `real.cuh` | done |
| 3 | Random numbers: Philox | `rng.cuh` | done (the *how* of the rounds: to revisit); C++ syntax and headers in [cpp_for_python.md](cpp_for_python.md) |
| 4 | Chunks: dividing the work | `main.cu` | done |
| 5 | The Scene: the star, prepared once | `scene.cuh`, `scene.cu` | done |
| 6 | One photon: the physics | `photon.cuh` | done |
| 7 | The image and the timings | `image.cu`, `main.cu` | done |
| 8 | The command line | `options.cu` | done |
| 9 | The driver and the three checks | `star.py`, `analysis.py` | done |
| 10 | A run with passes, from start to finish | `main.cu` | done |
| 11 | Observers: inclination → μ bin → file name; why `nmu` is odd | `observers.cu`, `options.cu` | done |
| 12 | Choosing K: observers per pass and the memory budget | `options.cu` | done |
| 13 | The image per pass: layout, zeroing, saving one observer, `.partial` then rename | `image.cu`, `files.cuh`, `photon.cuh` | done |
| 14 | What the engine tells: `PROGRESS`, `PASS`, `SUMMARY`, the records | `report.cu`, `run.cuh` | done |
| 15 | The driver's side: reading records, the passes check | `star.py`, `analysis.py` | done |
| — | **Recap of pieces 10–15: what we learned, what to change before the GPU** | | done |
| 16 | The processor split: the pass loop stops knowing which processor traces | `processor.cuh`, `cpu.cu`, `main.cu` | done |
| 17 | Random numbers on the GPU | `gpu_rng.cuh` | done |
| 18 | The star on the GPU: constant Scene, `GpuImage`, the kernel and its queue | `gpu.cu`, `gpu_image.cu` | done |
| 19 | The SM mask | `gpu_sms.cu` | done |
| 19b | ~~The CPU's cores~~ (deferred) | `cpu_cores.cu` | — |
| 20 | Memory, passes and the kernel's resources | `gpu_device.cu`, `gpu.cu`, `options.cu`, `report.cu` | done |
| 21 | The code mapped to the hardware, measured | `instruction_mix.py`, `kernel_experiments.py`, `gpu_monitor.py`, `gpu_sections.cuh` | done |
| 22 | The sweeps | `sweeps.py` | done |
| 23 | The scaling analysis | `scaling.py` | done |
| 24 | The showcase | `showcase.py` | done |
| 25 | **The products:** one entrypoint, `test` / `make` / `measure` / `compare`, the minimum N, the report | `main.py`, `lab/` | done |

**Pieces 1–24 name the files as they were then** (one engine folder, one script per job). Where everything lives now, and which command replaced which script: [piece 25's map](#the-map-from-the-old-names).

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
typedef float real;      // star_cpu32, star_gpu32  (built with -DSTAR_FP32)
#else
typedef double real;     // star_cpu64, star_gpu64
#endif
```

All the physics uses `real`, never `double` or `float` directly. The build compiles the same source twice (phase B: a Makefile; now `engine.py`). One copy of the physics, two precisions: the FP32 vs FP64 comparison compares exactly the same code.

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

---

## Piece 3: random numbers, Philox (`rng.cuh`)

### The big idea: a counter in, random numbers out

A classic generator (`rand()`, Mersenne Twister) has a **state** that changes at every draw: to get number 1 000 000 you compute the 999 999 before it. Philox is **counter-based**:

```
random block = philox4x32_10(counter, key)
```

- **`counter`:** 128 bits (four 32-bit words `x, y, z, w`). Says *which* block of numbers you want.
- **`key`:** 64 bits (`x, y`), the **seed**.
- **The output:** **four** random 32-bit numbers at once.

A pure function: the same counter and key always give the same four numbers. Any number of any stream is reachable directly by setting the counter. No history, no state shared between threads: exactly what thousands of GPU threads need.

### Inside: 10 rounds of scrambling

```c
static inline HD PhiloxCounter philox_round(PhiloxCounter counter, PhiloxKey key) {
    unsigned int high_0, high_1;
    unsigned int low_0 = multiply_high_low(PHILOX_ROUND_MULTIPLIER_0, counter.x, &high_0);
    unsigned int low_1 = multiply_high_low(PHILOX_ROUND_MULTIPLIER_1, counter.z, &high_1);
    PhiloxCounter mixed = {high_1 ^ counter.y ^ key.x, low_1, high_0 ^ counter.w ^ key.y, low_0};
    return mixed;
}
```

One round:

1. **Multiply** two of the four words by fixed odd constants. Each 32×32-bit product is 64 bits, split into a high and a low half. Multiplication spreads every input bit over many output bits.
2. **Mix** with XOR: the high halves with the other two words and the key.
3. **Shuffle:** the words come out in a new order, so the next round multiplies different words.

`philox4x32_10` repeats this **10 times**, adding a constant to the key between rounds (`0x9E3779B9`, from the golden ratio). After 10 rounds, flipping one bit of the counter flips about half of the output bits, unpredictably: that is what passing the statistical tests means. It is cheap on a GPU: the main cost is two multiplications per round, and the GPU gives the high half of a product in one instruction (`__umulhi`).

### The state: bookkeeping around the counter

```c
typedef struct {
    PhiloxCounter counter;       // which block we are at
    PhiloxCounter block;         // the four numbers of that block
    PhiloxKey key;               // the seed
    unsigned int next_in_block;  // which of the four to hand out next (0..3)
} PhiloxState;
```

`philox_next` hands out the four numbers of the current block one by one; after the fourth it adds 1 to the counter and computes the next block. A "stream" is just counting up.

### One stream per chunk: how the counter is split

```c
philox_init(seed, subsequence, offset, &state)   // same meaning as curand_init
```

- **The key** = the seed.
- **The counter's high half (`z, w`)** = the **subsequence**: our **chunk number**.
- **The counter's low half (`x, y`)** = the position inside that stream, counting up as numbers are drawn.

Chunk 17 lives in its own region of counter space, `(anything, 17)`.

- Each subsequence holds 2⁶⁴ blocks = 2⁶⁶ numbers, and a chunk of 1 024 photons uses about 6 000 of them: the streams never overlap.
- Setting one up costs nothing.
- Compare XORWOW, cuRAND's default: reaching subsequence 17 means *computing* a jump of 17 × 2⁶⁷ steps.

### A copy of cuRAND, checked

- **Every detail matches the CUDA 13.3 headers:** the constants, the round, the counter layout, the quirky carry in `philox_skip_blocks`, the conversion to a uniform number. The constants first kept cuRAND's names (`PHILOX_W32_0`, `PHILOX_M4x32_0`, …); in step 2 of the GPU, where `gpu.cu` includes both `rng.cuh` and `curand_kernel.h`, ours silently redefined cuRAND's macros (the compiler warned). They were renamed for what they are: `PHILOX_KEY_WEYL_0/1` (the key's increment per round: the golden ratio and √3 − 1 in 32 bits, a Weyl sequence) and `PHILOX_ROUND_MULTIPLIER_0/1`.
- **Checked now:** `./star --test-rng` against the Random123 known answers (3/3 pass).
- **To do, in the GPU step:** draw with real cuRAND on the GPU and compare with this copy, bit for bit.

### From 32 bits to a number in (0, 1]

```c
philox_next(state) * 2^-32 + 2^-32      // double, as curand_uniform_double
philox_next(state) * 2^-32 + 2^-33      // float,  as curand_uniform
```

- **One 32-bit draw per uniform number:** enough for the star ([precision.md §2](precision.md#2-random-bits-are-not-the-same-as-the-type)).
- **The interval is (0, 1]:** never 0, sometimes exactly 1. That matters if you ever take `log(ξ)`.

### Questions on piece 3

**What is new in the C++ here, coming from Python? Why do headers hold whole functions?**

Full answer: [cpp_for_python.md](cpp_for_python.md). In short:

- **`#include` is text copy-paste, not `import`.** Each `.cu` file is compiled alone, then the linker joins them.
- **Hot functions and future GPU code live in headers as `static inline`,** so the compiler sees their bodies and can inline them into the photon loop. That also satisfies `nvcc`, which by default needs device functions in the same file as the kernel.
- **Code that runs once is declared in a header and defined in a `.cu`** (`scene`, `image`, `options`).
- **Include guards** keep a header from being pasted twice into the same file.
- **How guards, `static inline`, templates and `HD` combine** on `trace_photon`: [cpp_for_python.md §6](cpp_for_python.md#6-putting-it-together-guards-static-inline-templates-and-hd). In short:
  - `nvcc` compiles each file twice, once for the CPU and once for the GPU.
  - `trace_photon` is a template so that the CPU version uses our Philox and the GPU version uses cuRAND, each calling only what exists on its side.

**How exactly do the rounds make the bits random?** To revisit later.

---

## Piece 4: chunks, dividing the work (`main.cu`, `options.cu`)

### How many chunks

```c
long long chunk_count(const Options *options) {
    return (options->star.photons + options->chunk_size - 1) / options->chunk_size;
}
```

- **Integer division rounding up:** N = 10⁸ with chunks of 1 024 gives 97 657 chunks.
- **The last chunk is partial:** 10⁸ = 97 656 × 1 024 + 256, so it holds only 256 photons.
- `(a + b - 1) / b` is the standard integer idiom for "a / b rounded up" (Python: `-(-a // b)` or `math.ceil(a / b)`).

### The loop over chunks: the sequential "queue"

```c
static PhotonCounts simulate_chunks(const Scene *scene, const Options *options, Image *image) {
    PhotonCounts counts = {0, 0, 0, 0};
    long long next_chunk = options->first_chunk;
    while (next_chunk < options->last_chunk) {
        simulate_chunk(scene, next_chunk, options, image, &counts);
        next_chunk++;
    }
    return counts;
}
```

- **It is written as a queue on purpose:** `next_chunk` is "the next chunk nobody took", a plain local variable, because there is one worker.
  - With CPU threads it would become a shared `atomic_fetch_add`.
  - On the GPU, a counter in device memory taken with `atomicAdd` (the work queue, [README §4](README.md#4-synchronization-atomics-and-the-work-queue)).
- **`[first_chunk, last_chunk)` is the chunk range.**
  - By default it is `[0, chunk_count)`, the whole run.
  - `--first-chunk` / `--last-chunk` select part of it; the two halves add up to the full image, bit for bit ([README §6](README.md#6-splitting-the-work-chunks)).
- **The counts are a struct returned by value**, and filled through a pointer (`&counts`) by each chunk. No globals.

### One chunk

```c
static void simulate_chunk(const Scene *scene, long long chunk, const Options *options,
                           Image *image, PhotonCounts *counts) {
    PhiloxState rng;
    philox_init(options->star.seed, (unsigned long long) chunk, 0, &rng);
    long long first_photon = chunk * options->chunk_size;
    long long end_photon = smaller(first_photon + options->chunk_size, options->star.photons);
    for (long long photon = first_photon; photon < end_photon; photon++) {
        long long pixel = trace_photon(scene, &rng);
        ...count it...
    }
    counts->simulated += end_photon - first_photon;
}
```

- **A fresh generator per chunk:** subsequence = chunk number. Chunk 17 draws the same numbers whoever runs it, whenever.
  - Inside the chunk, the photons are simulated **in order**, from that one stream.
  - That is why a photon may use a variable number of draws (one more in a penumbra) without breaking reproducibility: the order inside a chunk never changes.
- **`smaller(...)` clips the last chunk** at N.
- **`photon` only counts the iterations.** The physics does not depend on the photon's index, only on the stream's position; it is kept as `photon` for readability.
- **`image` can be `NULL`.** That is the `--no-record` mode: the same photons, the same counts, nothing written. It measures what recording costs.

### What changes on the GPU

| | CPU now | GPU next |
|---|---|---|
| who takes a chunk | one loop, `next_chunk++` | each thread, `atomicAdd(&next_chunk, 1)` on a counter in device memory |
| order of chunks | 0, 1, 2, ... | any order |
| generator | our `PhiloxState`, `philox_init(seed, chunk, 0)` | cuRAND, `curand_init(seed, chunk, 0)`: the same numbers |
| counts | `counts->recorded++` | in registers per thread, one `atomicAdd` per counter at the end |
| recording | `count_photon` → `counts[pixel]++` | `atomicAdd(&counts[pixel], 1)` |

The order of chunks does not matter because every chunk is self-contained and the image is integer sums.

### Choosing the chunk size (1 024)

- **Too small:** per-chunk overhead (initializing the generator, taking a chunk from the queue).
- **Too large:** small N does not give every GPU thread a chunk, plus a tail at the end where a few threads finish while the rest wait.
  - Laptop: ~55 000 threads, so N ≳ 6·10⁷ to give each one a chunk.
  - A30: ~115 000 threads, so N ≳ 10⁸.
- **It must stay fixed** across runs we compare: a different chunk size means different photons.

### Questions on piece 4

**Will `count_photon` handle both the host image and the device image?**

No. Today it is host-only (`counts[pixel]++` on RAM). We chose to keep the physics shared and give each side its own recording:

- **CPU:** `Image` + `count_photon`.
- **GPU:** `DeviceImage` + `count_photon_atomic` (`atomicAdd`), allocated with `cudaMalloc` and copied back at the end.

Separate types make the compiler reject a RAM image passed to the kernel; a single `HD` function branching on `__CUDA_ARCH__` would hide the atomic and allow that mistake. Details in [README §3](README.md#3-memory).

---

## Piece 5: the Scene, the star prepared once (`scene.cuh`, `scene.cu`)

### What it holds

```c
typedef struct {
    real linear_branch_probability;      // limb darkening: 3 / (3 + 2a)
    unsigned int spot_count;
    Vec3 spot_centers[MAX_SPOTS];        // unit vectors
    real umbra_chord_squared;            // spot core:   2 (1 - cos r_umbra)
    real penumbra_chord_squared;         // spot edge:   2 (1 - cos r_penumbra)
    real penumbra_escape_probability;    // 0.6, as in EP2
    unsigned int nx, ny, nphi, nmu, observers;
    int observer_of_imu[MAX_NMU];        // imu → observer, or NOT_AN_OBSERVER
} Scene;
```

Everything a photon needs to read, and nothing it writes.

- **No pointers, fixed size** (arrays sized by `MAX_SPOTS = 64` and `MAX_NMU = 4096`, ~17 KB).
  - On the GPU it is copied in one `cudaMemcpyToSymbol` into **constant memory** (64 KB, cached, broadcast when every thread of a warp reads the same address, which is what happens here).
  - A struct with pointers inside would need each pointed-to array copied separately, with GPU addresses fixed up by hand.
- **`StarConfig` vs `Scene`:** `StarConfig` is what the user asked for (radii in radians, `a`, counts); `Scene` is what the photon loop needs, already transformed (chords, probabilities, a lookup table).

### Computed once instead of once per photon

| EP2, per photon | EP4, once in `build_scene` |
|---|---|
| `cos(spot.inner_radius)`, `cos(spot.outter_radius)` for each spot | `umbra_chord_squared`, `penumbra_chord_squared` |
| `3.0 / (3.0 + 2.0 * a)` | `linear_branch_probability` |
| a loop searching the observer list for `imu` | `observer_of_imu[imu]`, one lookup |

### Spots by squared chord

For two unit vectors p and c at angle α: |p − c|² = 2(1 − cos α). So "angle < r" is the same as "|p − c|² < 2(1 − cos r)".

- **The threshold** 2(1 − cos r) is computed once.
- **Per photon** it is three subtractions, three multiplications, two additions: no `cos`, no `acos`.
- **Accurate in FP32:** the difference of two nearby vectors keeps its digits, while comparing two cosines near 1 loses them ([precision.md §4](precision.md#4-where-precision-actually-bites)).

### Placing the spots

```c
static int place_spots(const StarConfig *config, Scene *scene) {
    PhiloxState rng;
    philox_init(config->seed, SPOTS_SUBSEQUENCE, 0, &rng);
    for (unsigned int placed = 0; placed < config->spots; placed++) {
        ...
        do {
            candidate = random_point_on_sphere(&rng);
        } while (overlaps_placed_spots(candidate, scene, placed, config->penumbra_radius));
        scene->spot_centers[placed] = candidate;
    }
}
```

- **Their own random stream:** subsequence `SPOTS_SUBSEQUENCE = 2⁶⁴ − 1`, which no photon chunk ever reaches. Every run with the same seed, whatever its chunk range, places the same spots. That is what makes partial runs add up to the same star.
- **Rejection sampling:** draw a point, retry while it overlaps a placed spot (centers closer than twice the penumbra radius, as in EP2). A limit of 100 000 attempts stops impossible requests (too many or too large spots) with an error instead of looping forever.
- **In FP64 always**, even in the FP32 build: it runs once, on the CPU, so there is no reason to lose precision.

### The observer table

```c
observer_imu(observer) = observer * (nmu - 1) / (observers - 1)   // as in EP2: spread over imu
observer_of_imu[imu]   = observer, or NOT_AN_OBSERVER (-1)
```

- With 4 observers and nmu = 150: imu = 0, 49, 99, 149. Observers 0 and 3 are the poles (within ~9° of the axis), 1 and 2 at intermediate latitudes.
- **An error instead of a silent loss:** if two observers fall in the same μ bin (more observers than bins), EP2 silently gave one of them no photons; EP4 stops with a message.

### Validation

`validate_config` checks limits before anything is built: at most 64 spots, nmu ≤ 4096, 1 ≤ observers ≤ nmu, umbra ≤ penumbra. Functions return `0` or `-1` (the C convention), print why, and `main` exits with status 1.

---

## Piece 6: one photon, the physics (`photon.cuh`)

`trace_photon` is the professor's algorithm, step by step, each step a small function:

```c
template <typename Rng>
static inline HD long long trace_photon(const Scene *scene, Rng *rng) {
    SurfacePoint point = sample_surface_point(rng);                    // 1. where
    Vec3 direction = sample_emission_direction(scene, point, rng);     // 2. which way
    if (!escapes_spots(scene, point.position, rng)) {                  // 3. spots
        return PHOTON_BLOCKED_BY_SPOT;
    }
    ObserverBin bin = observer_bin(scene, direction);                  // 4. who sees it
    int observer = scene->observer_of_imu[bin.imu];
    if (observer == NOT_AN_OBSERVER) {
        return PHOTON_MISSES_OBSERVERS;
    }
    return (long long) pixel_index(scene, point.position, direction, bin, observer);   // 5. where on the image
}
```

Random draws, in order (the same order as EP2): z, φ, ξ_μ, ξ_branch, φ_local, and one more only inside a penumbra.

### 1. A point on the surface: `sample_surface_point`

```c
real z = 2 * uniform(rng) - 1;
real phi = TWO_PI * uniform(rng);
point.ring_radius = sine_from_cosine(z);     // r = sqrt(1 - z²)
point.position = (r cos φ, r sin φ, z)
```

- **Why z uniform and not the latitude.** On a sphere, equal steps in z cover equal areas (Archimedes: a sphere and its circumscribed cylinder have the same area between two heights). Drawing the latitude uniformly would crowd points at the poles.
- **`cos φ`, `sin φ` and `r` are kept** in `SurfacePoint`: step 2 needs them, and EP2 recomputed them (with an `atan2`) for every photon.

### 2. The emission direction: `sample_emission_mu` + `sample_emission_direction`

**μ = cos θ, the angle between the photon and the surface normal.**

- **The intensity** is I(μ) = I₀(1 + aμ) (limb darkening).
- **The flux through the surface** has one more factor μ (projected area), so p(μ) ∝ μ + aμ² on [0, 1].
- **The professor's slides** suggest inverting the cumulative distribution (a quadratic-cubic equation), or rejection. EP2 found a simpler way: p(μ) is a **mixture** of two densities that are easy to sample by inversion:

| Density | Normalized | Cumulative | Inversion | Weight |
|---|---|---|---|---|
| ∝ μ | 2μ | μ² | μ = √ξ | (1/2) / (1/2 + a/3) = 3 / (3 + 2a) |
| ∝ aμ² | 3μ² | μ³ | μ = ∛ξ | the rest |

```c
return xi_branch < scene->linear_branch_probability ? real_sqrt(xi_mu) : real_cbrt(xi_mu);
```

One uniform number picks the branch, another is inverted. With a = 1: 60% √ξ, 40% ∛ξ. Both are exact samples, no rejection.

**φ_local**, the azimuth around the normal, is uniform in [0, 2π].

**From the local frame to the star's frame.** (μ, φ_local) are angles relative to the surface normal at the point; the direction (u, v, w) must be in the star's (x, y, z). The rotation is EP2's (written out by hand, not a matrix):

```c
radial = sin θ · z · cos φ_local + μ · r
u = −sin θ · sin φ · sin φ_local + cos φ · radial
v =  sin θ · cos φ · sin φ_local + sin φ · radial
w =  μ · z − r · sin θ · cos φ_local
```

Check: with sin θ = 0 (emitted along the normal), (u, v, w) = μ(r cos φ, r sin φ, z) = the position, which is the outward normal of a unit sphere.

### 3. Spots: `escapes_spots`

```c
real distance = distance_squared(position, scene->spot_centers[i]);
if (distance < scene->umbra_chord_squared)    return false;                                   // core: no photon
if (distance < scene->penumbra_chord_squared) return uniform(rng) <= penumbra_escape_probability;  // edge: 60% escape
```

- Spots do not overlap, so at most one contains the point; the first match decides.
- The penumbra is a **probability**, not a dimmer intensity: each photon has a 60% chance to escape. Over many photons, the penumbra looks 60% as bright.

### 4. Who sees it: `observer_bin`

```c
azimuth = atan2(v, u), moved into [0, 2π)
iphi = bin_index(azimuth / 2π, nphi)
imu  = bin_index((w + 1) / 2, nmu)
```

- **Two binnings of the direction:** azimuth into nphi bins (the frames of the rotation video), μ = w into nmu bins (latitudes, equal solid angles).
- **`bin_index` is safe at the edges.**
  - A value just below 0 (rounding) or exactly 1 is clamped into the first or last bin, never out of the array.
  - `!(fraction > 0)` also catches NaN.
- **Checked before the pixel:** ~97% of photons fly toward no observer, so the expensive projection is skipped for them.

### 5. Where on the image: `pixel_index`

The professor's projection: rotate the star's frame about z by the observer's azimuth, then tilt it so x'' points at the observer. y'' and z'' are then the coordinates on the observer's sky (the impact parameter).

```c
root = sine_from_cosine(w)          // the professor's "raiz"
if (root == 0) { zpp = −x; ypp = y; }                 // looking straight down a pole
else {
    cf = u / root;  sf = v / root;                    // cos and sin of the observer's azimuth
    ypp = −sf·x + cf·y
    zpp = −(cf·x + sf·y)·w + z·root
}
ix = bin_index((ypp + 1) / 2, nx);   iy = bin_index((zpp + 1) / 2, ny);
pixel = ((iphi · observers + observer) · ny + iy) · nx + ix      // layout [iphi][observer][iy][ix]
```

- **The sky plane** of a unit star spans [−1, 1] in both axes, mapped to [0, 1) and then to pixels.
- **Near the poles,** `root` is small and dividing by it amplifies rounding. That is the FP32 hot spot we measured ([precision.md §8.5](precision.md#85-inside-the-star-does-it-matter)).
- **The layout:** all frames of one observer are interleaved by `iphi`, and each frame is one contiguous `ny × nx` picture.

### The return value

One `long long` carries everything:

- **≥ 0:** the pixel index;
- **`PHOTON_BLOCKED_BY_SPOT` (−1)** or **`PHOTON_MISSES_OBSERVERS` (−2)** otherwise.

The caller counts and records. `trace_photon` writes nothing, so the GPU can change how photons are recorded without touching the physics.

### Questions on piece 6

**What does "chord" mean in `umbra_chord_squared`?**

- **A chord** is the straight segment between two points on a circle or sphere, cutting through the inside, as opposed to the **arc**, which goes along the surface. The word comes from Greek *chordē*, a string, like a string stretched between two points on a bow.
- **For the spots:** the arc between the photon's point p and the spot center c is the angle α (on a unit sphere, arc length = angle). The chord is the straight distance |p − c|.
- **Arc and chord grow together,** so "angle < r" is the same test as "chord < chord of r", and the chord is much cheaper.
- **Chord squared, by the law of cosines** (two sides of length 1, angle α between them):

  |p − c|² = |p|² + |c|² − 2 p·c = 1 + 1 − 2 cos α = **2(1 − cos α)** = 4 sin²(α/2)

  So `umbra_chord_squared = 2(1 − cos r_umbra)` is the squared straight-line distance from a spot's center to the edge of its core.
- **Why squared:** it skips a square root per photon, and comparing squares gives the same answer as comparing distances (both are positive).

---

## Piece 7: the image and the timings (`image.cuh`, `image.cu`, `main.cu`)

### The image

```c
typedef struct {
    unsigned int *counts;    // one uint32 per pixel, in RAM
    size_t pixels;           // nx · ny · nphi · observers
} Image;
```

- **One flat array, not a 4D one.** C has no built-in multidimensional dynamic arrays. The four indices are folded into one by hand in `pixel_index`: `((iphi · observers + observer) · ny + iy) · nx + ix`. NumPy does the same internally, so `reshape(nphi, observers, ny, nx)` in Python reads the file back as 4D for free.
- **`size_t`:** the unsigned integer type for sizes and memory offsets (64 bits here). Sizes in bytes pass the 32-bit limit quickly: the Max image (1024², 360 frames, 4 observers) is 6 GB, more than a 32-bit `int` (2.1·10⁹) can hold.
- **`uint32` per pixel:** a pixel would need 4·10⁹ photons to overflow. EP2's default image is 96 MB; HD is 1.5 GB.

### Allocating: `malloc` + `memset`, not `calloc`

```c
image->counts = (unsigned int *) malloc(image_bytes(image));
memset(image->counts, 0, image_bytes(image));  // not calloc: zero every page now, inside the timed setup
```

- **`malloc(bytes)`** reserves memory without initializing it; `memset` fills it with zeros.
- **`calloc` would do both in one call**, and for big blocks it looks free. The operating system hands out pages that are only mapped and zeroed **the first time they are touched**.
  - So the real cost would move into the photon loop, as page faults the first time each page is written.
  - It would be counted as parallel time.
- **`memset` touches every page during setup,** where the cost belongs: serial time, for Amdahl's law.
- The one comment left in this file is there because this decision is invisible otherwise.

### Recording, saving, freeing

- **`count_photon(image, pixel)`:** `counts[pixel]++`. Inline in the header because it runs once per recorded photon. The GPU gets its own `count_photon_atomic` ([decision](README.md#3-memory)).
- **`save_image`:** `fwrite` of the raw array, no header and no format: just `pixels × 4` bytes. The configuration that gives it meaning goes in the `KEY=VALUE` line and in the driver's `run.json`. `fclose` is checked too, because buffered data is only flushed to disk there and a full disk shows up as an error at that point.
- **`free_image`:** gives the memory back and sets the pointer to `NULL`, so a second free or a late use fails clearly instead of silently.

### The timings

```c
static double now_in_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}
```

- **`CLOCK_MONOTONIC`, not the wall clock (`CLOCK_REALTIME`):** it never jumps backwards or forwards when the system adjusts its time (NTP, daylight saving). It is made for measuring intervals.
- **Read 6 times per run**, never per photon.

| Phase | Includes | Amdahl |
|---|---|---|
| `SETUP_TIME` | `build_scene` (spots, table) + allocating and zeroing the image | serial |
| `PHOTONS_TIME` | the chunk loop | parallel |
| `SAVE_TIME` | writing the image to disk | serial |
| `TIME` | everything from the start of `main`, including option parsing and freeing | total |

- **`PHOTONS_PER_SECOND` = photons / `PHOTONS_TIME`:** the throughput of the physics alone, the number to compare between EP2, the CPU, FP32/FP64 and the GPUs.
- **On the GPU, setup and save will grow:**
  - setup gains CUDA context creation (~0.15 s, measured with `nsys` on Buffon), `cudaMalloc` + `cudaMemset` of the image, and copying the Scene to constant memory;
  - save gains copying the image back (up to 1.5 GB for HD).
  - Those are exactly the serial terms the TCD's Amdahl analysis needs, already separated.

### `--no-record`

- `image` is `NULL`: no allocation, no `count_photon`, no file; `IMAGE=none`, `IMAGE_BYTES=0`.
- The same photons and the same counts, so **`PHOTONS_TIME` with and without `--no-record` measures what recording costs**.
- On the GPU, that is the experiment that answers whether the atomics matter. **[to run]**

---

## Piece 8: the command line (`options.cuh`, `options.cu`)

**The interface: one function.** `main` sees only:

```c
Options load_options(int argc, char *argv[]);
long long chunk_count(const Options *options);
```

Everything else in `options.cu` is `static`: private to the file. `main` does not know how parsing works, only what comes out of it.

### `argc` and `argv`

`int main(int argc, char *argv[])`: the program receives its command line as an array of C strings.

- `./star -n 1e8 --spots 0` gives `argc = 5` and `argv = {"./star", "-n", "1e8", "--spots", "0"}`.
- It is Python's `sys.argv`, with its length passed separately.

### `getopt_long`: the standard C parser

```c
while ((flag = getopt_long(argc, argv, "n:s:a:o:h", LONG_OPTIONS, NULL)) != -1) {
    read_option(&options, flag, optarg, argv[0]);
}
```

- **`"n:s:a:o:h"`:** the short options. A `:` after a letter means "takes a value" (`-n 1e8`); `h` takes none.
- **`LONG_OPTIONS`:** the `--long` names, each mapped to a code: a letter for the ones that also have a short form (`'n'`), or an `enum` value from 1000 up for long-only options (`OPT_SPOTS`, ...), so they never collide with letters.
- **Each call returns the next option's code,** with its value in the global `optarg`. It returns `-1` when there are no more options; `optind` is then the index of the first leftover argument (we reject leftovers).
- **Python equivalent:** `argparse`, but you write the loop and the `switch` yourself.

### Reading numbers: `number_or_stop`

```c
double value = strtod(text, &end);
if (end == text || *end != '\0' || value < smallest || value > largest) stop(message);
```

- **`strtod`** parses a double and sets `end` to the first character it did not use.
  - `end == text`: nothing was a number.
  - `*end != '\0'`: junk after the number (`"1e8x"`).
- **Parsing as a double** is what makes `-n 1e8` work, as in EP3. The cost: integers above ~9·10¹⁵ cannot be exact in a double, hence the limit on `-s`.
- **Every option has a range** and a message that says it. A bad value stops the program before anything runs.

### Defaults, and checks that need several options

- **`default_options()`** holds EP2's star: a = 1, 3 spots of 0.05 / 0.10 rad, 4 observers, 200×200, 150 × 150 bins, chunks of 1024, seed 42, output `star.bin`.
- **`validate_options`** runs after all options are read, for checks that combine them:
  - `-n` is required (except with `--test-rng`);
  - `--last-chunk` defaults to "all chunks" (`ALL_CHUNKS = -1` until N is known);
  - the chunk range must fit inside `[0, chunk_count]`.
- **Checks that need the star itself** (umbra ≤ penumbra, observers ≤ nmu, ...) are in `validate_config` in `scene.cu` (piece 5).

### `StarConfig` vs `Options`

- **`StarConfig`:** what the star *is* (N, seed, limb darkening, spots, observers, resolution). It goes to `build_scene`.
- **`Options` wraps it** with how to *run*: chunk size and range, record or not, output file, test mode.

---

## Piece 9: the driver and the three checks (`star.py`, `analysis.py`)

### The split: C computes, Python orchestrates and analyses

Same idea as EP1–EP3:

- **The engine** does one run and prints one `KEY=VALUE` line.
- **Python** compiles it, runs it as many times as an experiment needs, reads the images, and makes the pictures and reports.

The engine never knows about experiments; the driver never knows about photons.

### The engine, seen from Python

```python
compile_engine()                                  # make -s in EP4/engine (phase B; now engine.py calls nvcc)
result = run_engine(config, image_path, precision, chunk_range)
image = load_and_delete_image(result)             # numpy array [iphi][observer][iy][ix]
```

- **`run_engine`** builds the command line from a `config` dict, runs it with `subprocess.run`, and turns the last output line into a dict with `parse_key_values`. Each value is tried as `int`, then `float`, then kept as text.
- **`load_and_delete_image`** reads the raw `uint32` file with `np.fromfile`, reshapes it to 4D, and deletes it: images are large (96 MB here, 1.5 GB in HD); the pictures and reports stay.
- **Each experiment gets a folder** `runs/<date>_<time>-<name>/`.

### The four modes

`python star.py -n N [star options] [one or more checks]`. With no check, it simulates the star:

| Mode | What it runs | What it proves |
|---|---|---|
| *(default)* `simulate_star` | one run | a gif and a still frame per observer, plus `run.json` |
| `--check-split` | the full run, then the two halves of the chunk range | **the chunk range works:** halves add up to the full image, bit for bit |
| `--compare-ep2` | EP4 and EP2's own engine (compiled from `EP2/engine`), both without spots | **the physics is EP2's:** χ² between the images |
| `--compare-precision` | `star` (FP64) and `star32` (FP32), same seed | **how much FP32 changes:** the fraction of photons that land elsewhere, and the speed ratio |

### `--check-split`: exact equality

```python
halves = load_and_delete_image(first).astype(np.uint64) + load_and_delete_image(second)
identical = np.array_equal(halves, load_and_delete_image(full))
```

- No statistics: the same seed and chunks give the same photons, so the sum must be **exactly** the full image.
- The program exits with status 1 if it is not.

### `--compare-ep2`: two independent images of the same star

EP2 draws with `rand()`, so its photons are different ones; the images can only agree **statistically**.

- **No spots:** each engine would place its spots with its own random numbers, so the stars would differ.
- **Frames summed:** without spots, every frame shows the same star, so summing them gives 150× more counts per pixel.
- **The χ² for two Poisson counts a and b with the same mean:** (a − b)²/(a + b) averages ~1 per pixel. Only pixels with a + b ≥ 20 are used, where the approximation holds.
  - χ²/dof close to 1 means the same physics; "sigma from 1" = (χ² − dof)/√(2·dof) says how close.
  - Result: χ²/dof = 1.0047 over 126 008 pixels, +1.2σ.
- **`comparison.png`:**
  - both images;
  - the residuals (a − b)/√(a + b), which should look like noise;
  - their histogram against a standard normal.

### `--compare-precision`: the same photons in two precisions

Same seed, so FP32 and FP64 draw the same random numbers; any difference is rounding moving a photon across an edge.

- **`fraction_of_photons_moved`:** half the sum of |a − b| (each moved photon changes two pixels by 1), over the photons recorded. Result: 5·10⁻⁴.
- **Against the noise:** the relative Poisson noise per lit pixel, 1/√(mean count). Result: 0.22, hundreds of times larger.
- **A χ² is not used here:** it assumes independent images, and these share their photons.

### Pictures

- **`frame_to_picture`:**
  - scales counts to [0, 1] with the 99.9th percentile as white, so a few hot pixels do not wash out the rest;
  - applies the `hot` colormap;
  - flips rows so y points up.
- **`save_rotation_gif`:** all `nphi` frames of one observer, 80 ms each: the rotation video.

### Questions on piece 9

**Why does the driver delete the image?**

- **`load_and_delete_image`** reads the raw `.bin` into NumPy, then removes the file.
- **The images are large:** 96 MB at EP2's settings, 1.5 GB in HD, 6 GB at 1024². A sweep of dozens of runs would fill the disk with files the reports no longer need.
- **What stays** is what the image was for: the gifs and still frames, `run.json`, the reports and the comparison plots.
- `.gitignore` also excludes `*.bin`, so raw images never reach git.
- If a run's raw image is ever worth keeping (to re-analyse it later), the driver would need a `--keep-image` option. **[not needed yet]**

**How big can the image be, on the laptop and on the A30? How do we set the maximum?**

There is no maximum in the code; the real limits are GPU memory (to fit), photons and time (to fill). Full answer, with the numbers: [gpus.md, "How big can the image be?"](gpus.md#how-big-can-the-image-be). In short:

- **Laptop:** ~7 GB of GPU memory, largest ~1100² at 360 frames and 4 observers.
- **A30:** ~22 GB, largest ~1950².
- **Filling them takes 6·10¹² to 2·10¹³ photons,** hours at 10⁹ photons/s. Time, not memory, is the practical limit.

---

## Piece 10: a run with passes, from start to finish (`main.cu`)

Phase C's version of piece 1. The design is in [jobs_and_passes.md](jobs_and_passes.md); this is the code. Example: 4 observers, 2 per pass.

```
./star -n 1e7 --observers 1.5708,1.0472,0.5236,0 --observers-per-pass 2 -o out
```

```
main
 ├─ load_options          → K = 2 (or from --memory-budget), nmu = 47 from 150 frames
 ├─ prepare_run           SETUP
 │    ├─ build_scene        the star: spots, limb darkening (no observers yet)
 │    ├─ locate_observers   each inclination → μ bin → file name; duplicates are an error
 │    └─ allocate_image     room for K = 2 observers, not 4
 │
 ├─ for each pass (2 of them)
 │    run_pass
 │     ├─ assign_pass_observers   the Scene learns "this pass records bins 23 and 35"
 │     ├─ zero_image              ZERO
 │     ├─ simulate_chunks         PHOTONS: all 10⁷ photons, the same ones in every pass
 │     ├─ save_pass_observers     SAVE: per observer, the .bin and then its .record
 │     └─ print_pass              the PASS line
 │    add_pass                    adds this pass's counts and times to the run's totals
 │
 └─ print_summary         the SUMMARY line
```

### The Scene changes between passes, nothing else does

- `build_scene` makes the star once.
- Before each pass, `assign_pass_observers` fills `slot_of_imu`: for each μ bin, the slot of the image it is recorded in, or `NOT_IN_PASS`.

  | | bin 23 | bin 35 | bin 43 | bin 46 | every other bin |
  |---|---|---|---|---|---|
  | pass 1 | slot 0 | slot 1 | not in pass | not in pass | not in pass |
  | pass 2 | not in pass | not in pass | slot 0 | slot 1 | not in pass |

- **The photon loop is phase B's, unchanged.** A photon toward bin 43 in pass 1 is just `MISSED`.
- *Changes piece 5:* the Scene's `observer_of_imu` became `slot_of_imu`, filled per pass instead of once.

### The image is a set of slots, reused

- Allocated once, for **K observers** (`--observers-per-pass`, or as many as fit in `--memory-budget`).
- Each pass zeroes it and records its observers into the same slots. Memory stays K × footprint whatever the number of observers.
- An uneven split is fine: 4 observers with K = 3 gives a pass of 3 and a pass of 1. The last pass uses one slot.

### `add_pass`: a summary that does not depend on K

- Every pass traces the same photons, so `PHOTONS` and `BLOCKED` are the same each time: **copied, not added**.
- Each observer is recorded in exactly one pass, so `RECORDED` **adds up**.
- `MISSED` = photons − blocked − recorded.
- One pass or four, the `SUMMARY` counts are the same; only the times differ.

### A new phase: ZERO

- The image is zeroed at the start of **each pass** (`ZERO_TIME`), not once in SETUP as in phase B.
- SETUP is now the scene, the observers and the allocation (plus reading the options).
- *Changes piece 7:* the serial part of a run is `SETUP + ZERO + SAVE`; zero and save are paid once per pass.

---

## Piece 11: observers, from inclination to μ bin to file name (`observers.cu`, `options.cu`)

*Changes piece 5 and piece 8:* observers were a count spread over the μ bins (`observer · (nmu − 1)/(observers − 1)`); now they are a list of inclinations in radians.

### Reading the list (`read_inclinations`)

- `--observers 1.5708,1.0472,0.5236,0` is read number by number with `strtod`, the same function `number_or_stop` uses.
- Each value must be in [0, π]. The check allows up to 3.1416 so a typed π passes, then clamps to π.
- The list keeps its order: observer 0 is the first one typed. That index goes into the record (`OBSERVER`).

### Inclination → μ bin (`mu_bin_of`)

```
μ = cos i          fraction = (μ + 1) / 2          bin = ⌊fraction · nmu⌋   (clamped to nmu − 1)
```

- **The same formula as the photon's** (`observer_bin`, piece 6): a photon with direction z falls in bin ⌊(z + 1)/2 · nmu⌋. An observer at inclination i collects the photons whose z = cos i falls in its bin.
- **The clamp is for i = 0:** μ = 1 gives a fraction of 1, so bin `nmu`, one past the end. It becomes `nmu − 1`.
- **Why bins in μ, not in i:** equal widths in μ = cos i are equal solid angles (dΩ = dμ dφ), so every observer gets the same share of photons whatever its inclination. Archimedes' hat-box theorem again, the fact behind choosing z uniformly in `sample_surface_point`.

### The bin center and the file name (`describe_observer`)

```
μ_center = −1 + (bin + ½) · 2/nmu          i_center = acos(μ_center)
file stem: observer_mu0035_i1.0349
```

The default observers with nmu = 47 (computed):

| Requested i | fraction · nmu | Bin | Center i | Bin spans i |
|---|---|---|---|---|
| 1.5708 | 23.500 | 23 | **1.5708** | 1.5495 … 1.5921 |
| 1.0472 | 35.250 | 35 | 1.0349 | 1.0099 … 1.0594 |
| 0.5236 | 43.852 | 43 | 0.5528 | 0.5108 … 0.5921 |
| 0 | 47.000 | 46 | 0.2067 | 0 … 0.2928 |

- **An observer's image is an average over its whole bin,** not the view from exactly the typed inclination: 60° (1.0472) is a bin centered at 59.3° (1.0349).
- **The name carries the bin center,** so 1.0472 and 1.04 give the same bin and the same file. The typed value is kept in the record (`INCLINATION_REQUESTED`).
- **Two observers in one bin are an error:** `locate_observers` keeps `observer_in_bin`, and 1.5708 with 1.58 gives "fall in the same mu bin (23)". They would be the same image twice.

### Why `nmu` became odd (`nmu_from_frames`)

The same observers with nmu = 48 (frames/π rounded, the first rule):

| Requested i | fraction · nmu | Bin | Center i | Bin spans i |
|---|---|---|---|---|
| 1.5708 | **24.000** | 23 | 1.5916 | 1.5708 … 1.6125 |
| 1.0472 | **36.000** | 35 | 1.0711 | 1.0472 … 1.0947 |

- **With an even `nmu`, the equator (μ = 0) is an edge** between bins 23 and 24. 1.5708 is a hair above π/2, so μ is a hair below 0 and lands in the bin below, centered 1.2° off equator-on, with the observer at the very edge of its bin.
- **The same happens to any inclination on an edge:** 60° is one whenever 0.75 · nmu is a whole number (48 is).
- **With an odd `nmu`,** the middle bin (nmu − 1)/2 is centered on μ = 0: the equator-on observer is exactly equator-on. Rule: `nearest_odd(frames/π)`: 150 → 47, 300 → 95, 360 → 115.
- **The poles are always edges** (μ = ±1): a pole-on observer's bin center is at i ≈ √(2/nmu), 0.21 rad with nmu = 47. Near the pole a bin is narrow in μ but wide in angle.

---

## Piece 12: choosing K, the observers per pass (`options.cu`)

K is decided once, at the end of `validate_options`, by `choose_observers_per_pass`. The image size and the number of passes follow from it.

### The rules, in order

```
both --observers-per-pass and --memory-budget  → error: use one, not both
--no-record                                    → K = all (no image, nothing to split)
--memory-budget B                              → K = ⌊B / footprint⌋, at most all; 0 is an error
--observers-per-pass K                         → K, at most all
neither                                        → K = all: one pass, as in phase B
```

- `pass_count` = ⌈observers / K⌉.
- **The footprint** is one observer's image: `pixels_per_observer` × 4 bytes = size² × frames × 4. 24 MB at the defaults (200², 150 frames).

### Measured, with the default 4 observers

| Options | K | Passes | Image allocated |
|---|---|---|---|
| (none) | 4 | 1 | 96 MB |
| `--observers-per-pass 3` | 3 | 2 | 72 MB |
| `--observers-per-pass 9` | 4 | 1 | 96 MB (capped at all) |
| `--memory-budget 50MB` (or `0.05G`, `5e7`) | 2 | 2 | 48 MB |
| `--memory-budget 1GB` | 4 | 1 | 96 MB |
| `--memory-budget 1GB --no-record` | 4 | 1 | none |
| `--memory-budget 1MB` | error: one observer needs 0.024 GB | | |
| `--memory-budget 50MB --observers-per-pass 2` | error: use one, not both | | |

### Details

- **The budget is for the image only:** not the program, the CUDA context, or Python's copy. The headroom is the caller's job; the driver will compute the budget from the machine (~7 GB on the laptop's GPU, ~22 GB on the A30's, [gpus.md](gpus.md)).
- **Decimal units** (`bytes_per_unit`): `7GB` = 7·10⁹ bytes, as datasheets and disks count; case-insensitive; the `B` is optional.
- **K is a ceiling:** the engine never allocates more than K × footprint.

### Found on the way (in the [roadmap's backlog](roadmap.md#4-backlog-tasks-to-do))

- **Unbalanced passes:** K = 3 with 4 observers gives 3 + 1. Splitting 2 + 2 takes the same 2 passes (the same time) with 48 MB instead of 72. Fix: K′ = ⌈observers / passes⌉ once the passes are known.
- **Binary units:** `nvidia-smi` reports MiB, but `7GiB` is rejected today.

---

## Piece 13: the image in a pass (`image.cu`, `files.cuh`, `photon.cuh`)

### The layout: `[observer][frame][iy][ix]`

In `pixel_index`:

```
frame = slot · nphi + iphi
pixel = (frame · size + iy) · size + ix
```

Each observer is **one contiguous block** of nphi × size² counts, frames in order:

```
| slot 0: frame 0, frame 1, … frame 149 | slot 1: frame 0, … | …
```

*Changes piece 7:* phase B used EP2's `[iphi][observer][iy][ix]`, with the observers interleaved inside each frame (one observer in 150 separate pieces). Contiguous blocks give:

- **saving one observer** with one `fwrite`: the file *is* the block;
- **zeroing only the slots in use** with one `memset`;
- **on the GPU, copying one observer back** with one `cudaMemcpy` instead of a strided copy.

### Zeroing, and first touch (measured)

`allocate_image` is now only `malloc`; `zero_image` runs at the start of every pass. 512², 300 frames (315 MB per observer), 4 passes of 1 observer, two runs:

| | Pass 1 | Passes 2–4 |
|---|---|---|
| `ZERO_TIME` | **0.15–0.18 s** | 0.005–0.007 s |

- **25–30× slower in pass 1.** `malloc` only reserves addresses; the OS maps a real page the first time it is touched. Pass 1's `memset` pays for mapping ~77 000 pages (~2 GB/s); later passes write over mapped memory (~50 GB/s).
- That is the "first touch" the phase-B comment about `calloc` was about, now visible but mixed into pass 1's `ZERO_TIME` (backlog: report it apart, or touch the pages in setup).
- **On the GPU,** `cudaMalloc` maps the memory when it allocates, so this cost moves to SETUP.

### Saving one observer, safely (`save_observer`, `files.cuh`)

```
write the block to   observer_mu0047_i1.5708.bin.partial
check fwrite wrote everything, and that fclose succeeded
rename .partial  →   observer_mu0047_i1.5708.bin
```

- **`rename` is atomic** on one filesystem: there is either no `.bin` or the whole one. A run killed while writing leaves only a `.partial`, which the driver redoes.
- **The record follows the same steps, after the image:** a `.record` means its `.bin` is complete. It is the "done" marker.
- `rename` replaces an existing file atomically too, so a rerun never leaves a mix of old and new.
- **Limit:** without `fsync`, the data may still be in the OS's memory when the rename happens. Safe against a crash of the program, not against a power cut. `SAVE_TIME` (~0.15–0.18 s for 315 MB, ~2 GB/s) is the time to the OS's cache, not to the disk (backlog: decide on `fsync`).
- `PARTIAL_PATH_LENGTH` = `PATH_LENGTH` + the length of `".partial"`, because GCC warned that a 4096-byte path plus the suffix might not fit.

### Counting the observer's photons

- `photons_in_observer` sums the block after saving: `RECORDED` in the record, timed inside SAVE.
- A sanity number: the records of one pass add up to that pass's `RECORDED`.

### Questions on piece 13

**What is `fsync`? Is it like `rsync`?**

No; only the names are alike.

- **`rsync`** is a program that copies files between folders or machines, sending only what changed. We will use it in phase E to bring results back from IAG.
- **`fsync`** ("file sync") is a system call, a request to the operating system: *"put this file's data on the disk now, and don't return until it is there."*

Why it is needed: writing is two steps.

```
fwrite / write     →   the OS's page cache (RAM)     →   the disk
   returns here ↑         kept "dirty" for a while         written later, by the OS
```

- **`fwrite` and `fclose` return as soon as the data is in the OS's memory** (the page cache). The OS writes it to the disk later, by itself: on Linux, data older than 30 s (`dirty_expire_centisecs` = 3000) is written out.
- **If the program crashes,** nothing is lost: the data is in the OS, which will write it.
- **If the power goes** (or WSL or Windows is killed) before the OS writes it, the data is lost, but the rename may already be on the disk. The result: a `.bin` that looks finished and is not.
- **`fsync(fd)` before the rename** closes that gap: the rename happens only after the data is on the disk.

Measured on the laptop (315 MB, `dd`, two runs):

| | Run 1 | Run 2 |
|---|---|---|
| without `fsync` (to the cache) | 0.164 s (2.0 GB/s) | 0.080 s (4.1 GB/s) |
| with `fsync` (to the disk) | 0.184 s (1.8 GB/s) | 0.146 s (2.3 GB/s) |

- **The cost is small here:** a fast NVMe disk; tens of milliseconds per 315 MB. In a run of hours, nothing.
- **Correction, measured in the engine after adding `fsync`:** 0.31–0.43 s per 315 MB (~0.8–1 GB/s). The `dd` test overwrote the same file each time; the engine writes new files, which costs more on WSL's virtual disk. Still small in a run of hours, but visible in short checks.
- **`SAVE_TIME` without `fsync` measures the cache, not the disk,** so it varies with how busy the cache is (0.08 vs 0.16 s above). With `fsync` it measures the real save.
- **To be complete,** the folder itself is `fsync`ed after the rename, so the new name is on the disk too.

---

## Piece 14: what the engine tells the outside world (`report.cu`, `run.cuh`)

*Changes piece 1 and piece 7:* one result line became three kinds of line, plus a record per observer.

### One format: a kind, then `KEY=VALUE`

```
PROGRESS PASS=1 PASSES=1 CHUNK=98627 CHUNKS=244141 PHOTONS_PER_SECOND=1.009940e+07 ELAPSED=10.0 ETA=14.8
PASS PASS=2 PASSES=4 OBSERVERS=1 MU_BINS=35 PHOTONS=10000000 BLOCKED=41308 MISSED=9752155 RECORDED=206537 ZERO_TIME=0.000966 PHOTONS_TIME=1.001307 SAVE_TIME=0.018361
SUMMARY DEVICE=CPU PRECISION=64 N=10000000 … PASSES=4 … RECORDED=825903 … PHOTONS_PER_SECOND=1.012762e+07 OUTPUT_DIR=…
```

- **One parser for all:** Python's `parse_key_values` skips the first word; `lines_of_kind(output, "SUMMARY")` picks the lines wanted. `grep ^PASS` works too.
- **`setvbuf(stdout, NULL, _IOLBF, 0)`,** first line of `main`: into a pipe or a file, C buffers ~4 KB before writing, so progress would arrive in bursts or at the end. Line buffering sends each line when it is complete. That is what will make `status` live.

### `PROGRESS`: cheap on purpose

- `report_progress_if_due` runs after **every chunk** and prints only if 10 s have passed. Cost: one clock read (~25 ns) per chunk of 1024 photons (~100 µs): ~0.03%.
- **ETA** = (chunks left in this pass + passes left × chunks per pass) ÷ this pass's chunk rate. Measured: at 10 s it said 14.8 s to go, at 20 s 4.1 s; the run ended at ~24 s.
- It leaves out zero and save times (small next to the photons). The rate restarts each pass, so each pass's first line comes 10 s in.

### `PASS` and `SUMMARY`

- `PASS`: the pass's μ bins, its counts, its three times.
- `SUMMARY`: the configuration, the counts (independent of K, piece 10), all phase times.
- **`PHOTONS_PER_SECOND` counts traced photons** (photons × passes ÷ photon time): the tracing speed, comparable across K (4 passes of 10⁷: 1.01·10⁷/s, as one pass).

### The record: one per observer, one `KEY=VALUE` per line

```
FINGERPRINT=none   CODE_HASH=none                      ← passed in by the driver
IMAGE=observer_mu0047_i0.2045.bin   OBSERVER=3   MU_BIN=47
INCLINATION_REQUESTED=0   INCLINATION_BIN_CENTER=0.204480199
DEVICE=CPU   PRECISION=64   N=…  SEED=…  A=…  SPOTS=…  SIZE=…  FRAMES=…  NMU=…
CHUNK_SIZE=…  FIRST_CHUNK=…  LAST_CHUNK=…
PASS=1   PASSES=1   PASS_OBSERVERS=4
PHOTONS=10000000   BLOCKED=41308   RECORDED=205525   IMAGE_BYTES=24000000
PHOTONS_TIME=1.133055   SAVE_TIME=0.019488   FINISHED=2026-11-21T19:11:03Z
```

(one key per line in the file; grouped here to fit)

- **Everything needed to read and trust the `.bin` alone:** its shape (SIZE, FRAMES), what it is (μ bin, inclinations, precision, chunk range), where and when it was made. Two images are comparable only if their records agree on these.
- **`PHOTONS_TIME` belongs to the pass,** shared by its `PASS_OBSERVERS` observers. Summing it over a pass's records would count it several times; `PASS_OBSERVERS` is there to say so.
- **Two clocks:** durations use `CLOCK_MONOTONIC` (`now_in_seconds` in `run.cuh`), which never jumps when the system clock is corrected; `FINISHED` uses wall-clock UTC (`time` + `gmtime_r`), because it must be a date.

### `run.cuh`: the vocabulary

- Small structs named for what they hold: `PhotonCounts`, `PassTimes`, `PassResult` (one pass's counts and times), `ObserverResult` (one observer's photons, bytes, save time), `RunTimes`, `Run` (options, observers, passes, start time), `Progress`.
- `report.cu` only formats them; it decides nothing.

### Found on the way (in the backlog)

- **No machine name:** the laptop's CPU and the A30 server's CPU both say `DEVICE=CPU`. Add `HOST=`.
- **No spaces in values:** an `OUTPUT_DIR` or `FINGERPRINT` with a space would break the split. State it, or reject spaces.

---

## Piece 15: the driver's side (`star.py`, `analysis.py`)

*Changes piece 9:* the same small driver, adapted to the new engine. It is also the first bit of phase D, since it already reads records.

### Talking to the engine

- **`engine_command`:** observers as `",".join(f"{i:.17g}")`, 17 significant digits so a double survives the trip to C exactly. `--nmu` and `--observers-per-pass` only when set, so the engine's defaults (odd nmu, one pass) apply otherwise.
- **`run_engine`** takes an **output folder** instead of an image file and returns the `SUMMARY` line (`lines_of_kind`). `parse_key_values` skips words without `=`, which drops the line kind.

### Reading the results: the records are the index

```python
records = sorted(read_record(path) for path in glob("*.record"), key=OBSERVER)
image   = np.stack([load_observer_image(record["IMAGE"], record["FRAMES"], record["SIZE"]) ...])
```

- **Python looks for `.record` files, never `.bin`:** a `.bin` without its record (an interrupted save) is ignored. Piece 13's "done marker" rule, already used in miniature; phase D's job state is the same idea.
- **The record says how to read the `.bin`:** `FRAMES` and `SIZE` give its shape.
- **Sorting by `OBSERVER`** restores the typed order (file names sort by μ bin).
- The result is `[observer][frame][y][x]`: summing frames moved from `axis=0` to `axis=1`; photons per observer is `sum(axis=(1, 2, 3))`.

### The checks

| Check | Runs | Passes if | Today |
|---|---|---|---|
| **`--check-passes`** (new) | all observers in 1 pass, then `--observers-per-pass 1` | images `array_equal` and `PHOTONS`, `BLOCKED`, `MISSED`, `RECORDED` equal | identical |
| `--check-split` | the full run, then two halves of the chunk range | halves summed (`uint64`) = full | identical |
| `--compare-ep2` | EP4 and EP2, no spots | χ² between the images, frames summed | 0.9957 |
| `--compare-precision` | FP64 and FP32 builds | photons moved vs Poisson noise | 1.2·10⁻⁴ |

- **EP2's observers:** EP2 places them by count, at bins o · (nmu − 1)/(count − 1) (0, 15, 30, 46 for nmu = 47). `ep2_observer_inclinations` gives EP4 the **bin centers** of those bins (2.935, 1.918, 1.268, 0.207 rad), so `locate_observers` lands in the same bins.
- `load_ep2_image` transposes EP2's `[frame][observer][y][x]` to the new order.

### What is kept

- **Checks:** `load_and_delete_observers` removes the output folder; `report.txt` (and the plot) stay.
- **A run:** gifs named by the file stem (`observer_mu0023_i1.5708.gif`); the `.bin` files are deleted, the records kept.

### Found on the way (in the backlog)

- **Everything is loaded at once:** all observers stacked in RAM, then converted to float64 for the pictures. Fine at 24 MB per observer; impossible at 2k (5 GB per observer, 40 GB as float64). Read one `.bin` at a time with `np.memmap`.
- **`nmu` is derived twice,** in C and in Python (`ep2_nmu`). Two copies of a rule can drift apart.

---

## Recap of pieces 10–15: what we learned

### About passes

1. **A pass is a full re-trace of the same photons.** Only the Scene's `slot_of_imu` changes between passes; the photon loop is phase B's.
2. **Cost ≈ passes × trace time.** The trace dominates; computing the pixel for 8% or 2% of the photons makes no measurable difference; writing to the image costs ~180 ns per recorded photon. *(My first explanation, "the smaller image fits the cache", was wrong.)*
3. **The summary does not depend on K:** photons and blocked are copied, recorded adds up, missed follows.
4. **K is a ceiling** from `--observers-per-pass` or `--memory-budget`; the budget covers the image only.

### About observers

5. **μ bins are equal solid angles**, so every observer gets the same share of photons.
6. **An observer's image is an average over its bin;** the file name carries the bin center, the record the typed inclination.
7. **`nmu` must be odd** for the equator to be a bin center; the poles are always bin edges (pole-on is really i ≈ √(2/nmu)).

### About memory and files

8. **One contiguous block per observer** (`[observer][frame][iy][ix]`): one `fwrite`, one `memset`, and on the GPU one `cudaMemcpy`.
9. **First touch:** `malloc` only reserves; pass 1's zeroing pays for mapping the pages (25–30× slower than later passes).
10. **Write `.partial`, then rename:** a file that exists is whole; the record, written last, is the "done" marker. Without `fsync`, safe against a crash, not a power cut.

### About reporting

11. **A kind, then `KEY=VALUE`; line-buffered** so the driver sees each line live. Progress costs ~0.03%.
12. **Records are self-describing:** everything needed to read and trust a `.bin` alone. `PHOTONS_TIME` belongs to the pass, not the observer.
13. **Python finds results through the records,** never the `.bin` files.

### What this means for the GPU engine

- **Per pass:** copy the 16 KB `slot_of_imu` table into constant memory (`cudaMemcpyToSymbol`), `cudaMemset` the slots in use, launch, then copy each observer's block back.
- **The host needs room for one observer, not K:** copy one block back, write it, reuse the buffer.
- **The trace is what parallelizes;** the writes become atomics, and their cost is what Nsight Compute will show.
- **First touch moves to `cudaMalloc`,** so the CPU and GPU zero times should mean the same thing (see below).
- **Progress while the kernel runs** is an open question: the host must read the chunk counter from the device without stopping the kernel.

### To change before the GPU

The engine changes that the GPU engine will inherit. Cheaper to do once, now. Tracked in the [roadmap's backlog](roadmap.md#c-before-the-gpu-engine-changes-the-gpu-will-inherit).

---

## Piece 16: the processor split: the pass loop stops knowing which processor traces

`processor.cuh`, `cpu.cu`, `gpu.cu`, `gpu_device.cu`, `main.cu`, `engine.py`. Design: [gpu_engine.md](gpu_engine.md), [hpc_architecture.md](hpc_architecture.md).

### What a processor must do (`processor.cuh`)

```c
typedef struct Processor Processor;                     // opaque: each processor defines it

void describe_processor(const Options *, ProcessorInfo *);   // kind, name, compute capability, compute units
int  compare_rng_with_cpu(unsigned long long seed);          // the GPU checks cuRAND against rng.cuh (step 2)
Processor *open_processor(const Options *);                   // allocate the image where this processor writes
void load_pass(Processor *, const Scene *);                  // the CPU keeps a pointer; the GPU copies to constant memory
void zero_pass_image(Processor *, unsigned int observers);
PhotonCounts trace_pass(Processor *, const Run *, Progress *);
const PixelCount *observer_on_host(Processor *, unsigned int slot);   // the CPU: a pointer; the GPU: a copy back
size_t processor_image_bytes(const Processor *);
void close_processor(Processor *);
```

- **An opaque type** (`typedef struct Processor Processor;` with no body in the header): `main.cu` can hold a `Processor *` and pass it around, but cannot look inside. `cpu.cu` defines `struct Processor { options, scene, record, Image image; }`; `gpu.cu` will define its own, with device pointers. In Python terms: an object whose attributes are private, used only through its methods.
- **Chosen at link time:** both files define the same functions. `star_cpu64` is linked with `cpu.cu`, `star_gpu64` with `gpu.cu`; the linker connects `main.cu`'s calls to whichever is there ([cpp_for_python.md §1](cpp_for_python.md)). No `#ifdef`, no `if (gpu)` in the pass loop.

### The pass loop now (`main.cu`)

```
run_pass:
    assign_pass_observers(&scene, ...)        the Scene learns this pass's μ bins
    load_pass(processor, &scene)
    zero_pass_image(processor, ...)           ZERO
    trace_pass(processor, run, &progress)     PHOTONS
    for each observer:
        observer_on_host(processor, slot)     COPY  (new phase; ~0 on the CPU)
        save_counts, the record               SAVE  (shared by both processors)
```

- **`COPY` is new,** so the GPU's device → host copy is measured apart from the disk. On the CPU it is a pointer, ~0 s.
- **Saving takes a block of counts** (`save_counts(counts, pixels, path)`, `photons_in`), not an `Image`, so the CPU's image and the GPU's host buffer are saved by the same code.
- **`SAVE_TIME` includes the records** (measured as the whole loop minus the copies).

### The processor's description in the output

```
PROCESSOR=CPU PROCESSOR_NAME=Intel(R)_Core(TM)_Ultra_7_255HX COMPUTE_CAPABILITY=none COMPUTE_UNITS=1 UNIT=core
PROCESSOR=GPU PROCESSOR_NAME=NVIDIA_GeForce_RTX_5070_Laptop_GPU COMPUTE_CAPABILITY=12.0 COMPUTE_UNITS=36 UNIT=sm
```

- The CPU's name from `/proc/cpuinfo`; the GPU's from `cudaGetDeviceProperties`. Spaces become `_` (EP3's `replace_spaces_with_underscores`, now in `run.cuh`) so `KEY=VALUE` stays one word per value.
- `COMPUTE_UNITS` is what `scaling.py` will read, whichever the processor.
- `gpu_device.cu` holds `check` (stop with a message on any CUDA error), `gpu_properties`, `gpu_attribute`: EP3's helpers, same names.

### The build without a Makefile (`engine.py`)

```
nvcc -O2 -std=c++17 -Xcompiler -Wall -arch=native [-DSTAR_FP32] [-Xptxas -v]
     main.cu options.cu scene.cu observers.cu image.cu report.cu  cpu.cu | gpu.cu gpu_device.cu
     -o build/star_<processor><precision>
```

- **One call, three steps** (preprocess, compile each `.cu`, link), as in [cpp_for_python.md §1](cpp_for_python.md).
- **Rebuild only when needed:** the build hash (all the engine's sources + the command + `nvcc --version`) is stored in `build/<binary>.hash`. Measured: 4 s to build a CPU binary; 0.00 s when the hash matches.
- **The hash covers every engine file,** so changing `gpu.cu` also rebuilds the CPU binaries. Simple and safe: one code hash for the whole engine, the one the records carry (`CODE_HASH`).
- **`-Xptxas -v`** for the GPU binaries: the compiler's resource report (registers, stack, spills per kernel), kept in `build/<binary>.report`.
- **What the GPU binary contains:** `cuobjdump --list-elf --list-ptx` shows `sm_120` cubins only, no PTX (as decided, [gpus.md](gpus.md#what-the-code-must-do-to-scale-there-unchanged)).

### Checks

All four CPU checks, after the split: **the same numbers as before**: split and passes bit for bit, EP2 χ²/dof = 0.9957, FP32 moves 1.2·10⁻⁴ of the photons. The refactoring changed the structure, not the results.

---

## Piece 17: random numbers on the GPU (`gpu_rng.cuh`, `gpu.cu`)

### The generator the physics sees

```c
typedef curandStatePhilox4_32_10_t GpuRng;

static inline __device__ real uniform(GpuRng *state) {
#ifdef STAR_FP32
    return curand_uniform(state);            // x · 2⁻³² + 2⁻³³
#else
    return curand_uniform_double(state);     // x · 2⁻³² + 2⁻³²  (one 32-bit draw, for Philox)
#endif
}
```

- **The second `uniform`:** `rng.cuh` has `uniform(PhiloxState *)` for the CPU; `gpu_rng.cuh` adds `uniform(GpuRng *)` for the GPU. `trace_photon` is a template on the generator type (piece 6), so the compiler picks the right `uniform` for each: the same physics, two generators.
- **`__device__` only:** cuRAND's device API cannot run on the host, which is why our CPU copy exists (piece 3).
- **Header-only:** `curand_kernel.h` brings the generator's code into our file; nothing to link.

### The check: the same numbers on both, bit for bit

`--test-rng` on `star_gpu64` / `star_gpu32` launches a tiny kernel, `draw_first_numbers`, one thread per chunk checked:

```
curand_init(seed, chunk, 0, &rng)       chunk = subsequence, as on the CPU
16 × curand(&rng)                       the raw 32-bit numbers
curand_init again; 16 × uniform(&rng)   the numbers the physics uses (FP64 or FP32)
```

and compares each with `rng.cuh` on the host (the uniforms with `memcmp`, so bit for bit, not "close"):

```
GPU_SEED_42_CHUNK_0=9ceaf053,77f5493b,12bf50ad,5742b3d7,fcdb2127,53ba6cfd,838f5a6e,744e06fb
GPU_RNG_CHUNK=0 BITS=SAME_AS_CPU UNIFORMS=SAME_AS_CPU
GPU_RNG_CHUNK=1 BITS=SAME_AS_CPU UNIFORMS=SAME_AS_CPU
GPU_RNG_CHUNK=1000003 BITS=SAME_AS_CPU UNIFORMS=SAME_AS_CPU
GPU_RNG_CHUNK=4294967301 BITS=SAME_AS_CPU UNIFORMS=SAME_AS_CPU
```

- **Chunk 4 294 967 301 = 2³² + 5:** the subsequence no longer fits one 32-bit word of the counter, so it spills into the next one (`philox_skip_subsequences`). The same on both: the copy matches cuRAND even there.
- **The GPU's chunk 0 = `SEED_42_CHUNK_0` printed by the CPU** since phase B: the number we wrote down then is now confirmed on the hardware.
- **This is what makes "GPU = CPU" possible** (step 3): any difference left will come from the arithmetic, not from the random numbers.

### Found on the way

- **A macro collision:** our constants kept cuRAND's names; with both headers in `gpu.cu`, ours redefined cuRAND's (the compiler warned). Renamed `PHILOX_KEY_WEYL_0/1` and `PHILOX_ROUND_MULTIPLIER_0/1` (piece 3 updated).
- **218 048 bytes of global memory** in the compiler's report for `gpu.cu`: most likely cuRAND's precomputed tables for its *other* generators (XORWOW, MRG32k3a), which `curand_kernel.h` brings in even though Philox needs none. Philox is counter-based: it skips ahead by arithmetic, not with tables.
- **The test kernel:** 19 registers, no stack, no spills.

---

## Piece 18: the star on the GPU (`gpu.cu`, `gpu_image.cu`)

The plan: [gpu_engine.md §4](gpu_engine.md#4-a-pass-on-the-gpu). The queue is EP3's (tested there, [§8](gpu_engine.md#8-what-ep3-already-settled-review-of-the-ep3-sessions-2026-11-21)); what is new is the inside of a chunk and everything around the kernel.

### The kernel, top to bottom

```c
__constant__ Scene gpu_scene;                                   // the star, in constant memory

__global__ void trace_photons(const PhotonWork work, PixelCount *image, GpuCounters *counters) {
    add_tallies(counters, take_chunks_until_none_left(work, image, counters));
}
```

| Function | Does | As in |
|---|---|---|
| `take_next_chunk` | `atomicAdd(&counters->next_chunk, 1)`: the work queue, one hot address | EP3, unchanged |
| `take_chunks_until_none_left` | a persistent thread: take a chunk, trace it, take the next, until none left | EP3, unchanged |
| `trace_chunk` | `curand_init(seed, chunk, 0)`, then `trace_photon(&gpu_scene, &rng)` for each photon of the chunk; tally it; if recorded, `count_photon_atomic` | the CPU's `simulate_chunk`, line for line |
| `add_tallies` | the thread's four counts (registers), added once with `atomicAdd` | EP3's `crossings` |

- **`PhotonWork`** (seed, N, chunk size, chunk range) is passed by value: kernel arguments live in their own constant bank.
- **`GpuCounters`** (the queue's counter and the four tallies) is one small struct in device memory, zeroed with `cudaMemset` before each pass.
- **`trace_photon(&gpu_scene, &rng)`** is the same function the CPU calls (piece 6), with cuRAND's state instead of our copy (piece 17). The pointer to a `__constant__` variable, once inlined, reads from the constant cache.

### Around the kernel, per pass

| Phase | GPU | CPU |
|---|---|---|
| SETUP | `cudaMalloc` the image (K observers) + `cudaMemset` it once (the first touch); `cudaMallocHost` a **pinned** host buffer of **one** observer; `choose_launch` | `malloc` + `touch_every_page` |
| load the pass | `cudaMemcpyToSymbol(gpu_scene, &scene)`: ~18 KB into constant memory | keep a pointer |
| ZERO | `cudaMemset` of the slots in use | `memset` |
| PHOTONS | reset the counters, launch, `cudaDeviceSynchronize`, read the counters | the chunk loop |
| COPY | one `cudaMemcpy` device → pinned buffer, per observer | a pointer |
| SAVE | shared: `.partial`, `fsync`, rename, the record | shared |

- **Pinned memory** (`cudaMallocHost`): memory the OS may not move, so the GPU's copy engine can write into it directly (DMA). Copying into ordinary (pageable) memory goes through a hidden pinned buffer, one more copy. One observer's worth (24 MB here) is enough, since observers are copied one at a time.
- **The launch** (`choose_launch`, EP3's): the occupancy API picks the block size and the blocks per SM for `trace_photons`, and the grid is blocks per SM × 36 SMs. No `__launch_bounds__` (EP3 had 1024): the star's kernel needs more registers than Buffon's, so the compiler is left free and the occupancy API sizes the blocks around it. *(The numbers come in step 5.)*

### What the compiler says

| | FP64 | FP32 |
|---|---|---|
| registers per thread | 78 | 67 |
| stack | 40 bytes | 32 bytes |
| spills | 0 | 0 |

(EP3's kernel: 58 registers.) The stack is small local memory, most likely the thread's tallies or the `SurfacePoint`/direction structs; the instruction mix (step 6) will show what.

### GPU = CPU?

| N | FP64 | FP32 |
|---|---|---|
| 10⁷ | **identical, bit for bit** | 54 pixels differ: 3.2·10⁻⁵ of the photons |
| 10⁸ | **identical, bit for bit** | — |

- **The random numbers are identical** (piece 17), so every difference comes from arithmetic.
- **FP64: no photon changes pixel.** CUDA's FP64 `sin`, `cos`, `atan2`, `cbrt` and the fused multiply-adds round differently from glibc in the last bit, but at FP64's 53 bits a photon almost never sits close enough to a pixel edge for that to matter: none at 10⁸ photons.
- **FP32: 27 photons of 843 339 move.** Experiment: the same FP32 build with `--fmad=false` (no fused multiply-add) moves only 12 pixels (7.1·10⁻⁶). **FMA explains ~3/4 of the difference; the rest is CUDA's FP32 math library** (`sinf`, `cosf`, `atan2f`, `cbrtf` are not glibc's).
- **For scale:** FP32 vs FP64 moves 1.1·10⁻⁴; GPU vs CPU in FP32 moves 3.5× less. Both are far below the Poisson noise (0.12 per lit pixel).
- **For the class: bit for bit vs statistically equal.** Same random numbers + FP64 = the same image on two different processors; FP32 = statistically equal, and we can say why.

### The other checks, on the GPU

- split and passes: **bit for bit**, FP64 and FP32;
- EP2: χ²/dof = 0.9957, the same number as the CPU's (the images are identical);
- FP32 vs FP64 on the GPU: FP32 **14.6×** faster (the CPU: 1.45×), the same 1.1·10⁻⁴ moved.

### First speeds (laptop GPU; one run each; the sweeps will repeat)

**What is compared:** the **whole GPU (all 36 SMs)** against **one CPU core** (the CPU engine is single-threaded), and by default only **the photon phase** (`PHOTONS_PER_SECOND`: the kernel vs the chunk loop). Every speedup below says which.

| | N | PHOTONS phase | photons/s | photon phase vs one CPU core |
|---|---|---|---|---|
| GPU FP64 | 10⁸ | 0.158 s | 6.3·10⁸ | **75×** |
| GPU FP32 | 10⁸ | 0.011 s | 9.3·10⁹ | ~700× |
| GPU FP64 | 10⁷ | 0.026 s | 3.9·10⁸ | 41× |
| GPU FP32 | 10⁷ | 0.003 s | 3.8·10⁹ | 291× |

**The whole run** (setup + zero + photons + copy + save), from the same runs, approximately:

| | CPU, whole run | GPU, whole run | whole-run speedup | photon-phase speedup |
|---|---|---|---|---|
| FP64, N = 10⁷ | ~1.2 s | ~0.29 s | **~4×** | 41× |
| FP32, N = 10⁷ | ~1.0 s | ~0.37 s | **~3×** | 291× |
| FP64, N = 10⁸ | ~10.7 s | ~0.48 s | **~22×** | 75× |

- **The gap between the two columns is Amdahl's law:** the GPU pays ~0.15 s to create the CUDA context before any photon; the bigger N, the smaller that share.
- **Not against the whole CPU:** all 20 cores (8 P + 12 E) would narrow the gap, roughly by 10–15× (an estimate; CPU threads are deferred). Stated as "vs one core", following Hoefler and Belli ([hpc_architecture.md §4](hpc_architecture.md#4-benchmarking-like-hpc)).
- **Not per SM:** one SM vs one core would be ~75/36 ≈ 2× in FP64 and ~20× in FP32 *if* the GPU scaled perfectly. The SM mask (step 4) measures it instead of assuming it: see piece 19.
- **Amdahl, already visible:** at N = 10⁸ in FP32, the photons take 0.011 s and SETUP (mostly creating the CUDA context) 0.14 s; the run is 90% serial. At 10⁷ the GPU's speed looks lower because the kernel is too short to fill and drain the GPU efficiently. The `problem-size` sweep (step 7) measures exactly this.
- **FP32 is 14.6× faster than FP64 on this GPU** (its FP64 units run at 1:64), against 1.45× on the CPU: the same code, opposite conclusions about precision, as predicted in [gpus.md](gpus.md).

---

## Piece 19: the SM mask, and the work per SM (`gpu_sms.cu`)

EP3's design, now for the star: run on N SMs of the 36, to measure Amdahl's law over SMs.

### How the mask works

```
open_processor:
    choose_launch                      blocks per SM × 36 SMs: the grid covers the whole GPU
    create_sm_work(sms, launch)
        list_sm_ids: a tiny kernel (mark_sm_ids) with the same launch; each block writes its SM's id
        allow the first `sms` ids found   (the ids need not be 0…35: list them, don't assume)
trace_photons, first lines:
    sm = sm_id()                       the %smid register: which SM this thread runs on
    if (!is_allowed(sm)) return;       blocks on other SMs leave at once
```

- **Cooperative launch** (`cudaLaunchCooperativeKernel`, when the GPU supports it, as EP3): the launch fails unless every block can be resident at once. That is what makes the mask honest: all blocks start together, the ones on forbidden SMs leave, and the queue is shared only by the allowed SMs.
- **`--sms N`** on the GPU engines; the CPU engine refuses it ("--sms needs a GPU engine"). `COMPUTE_UNITS` in the output is the number of SMs allowed.

### The work per SM (`SmWork`, in device memory)

Each thread, before leaving, adds to its SM's counters: chunks taken, photons, recorded, and its cycles (`clock64()`, from its first line to its last; `atomicMax` keeps the slowest thread). After each pass:

```
SM PASS=1 SM=0 CHUNKS=27136 PHOTONS=27787264 RECORDED=2345585 CYCLES=251587808
...
SMS PASS=1 SMS_USED=36 CYCLES=251640725
```

**Measured (FP32, N = 10⁹, all SMs):** chunks per SM from 27 104 to 27 136 (0.1%), cycles from 251.2 M to 251.6 M (0.2%). **The queue balances the SMs almost perfectly:** a fast SM simply takes more chunks.

### Checks

- **The same image on 1, 2, 9 and 36 SMs** (FP32, N = 10⁸): the files are identical byte for byte. Who traces a chunk never changes its photons.
- FP64 kernel: 80 registers (78 before the mask).

### First scaling over SMs (FP32, N = 10⁸, photon phase, one run each)

| SMs | time | speedup over 1 SM | cycles of the slowest SM | implied clock = cycles / time |
|---|---|---|---|---|
| 1 | 0.313 s | 1 | 903 M | **2.88 GHz** |
| 2 | 0.157 s | 2.0 | 452 M | 2.88 GHz |
| 9 | 0.036 s | 8.7 | 103 M | 2.85 GHz |
| 36 | 0.015 s | 20.8 | 29.5 M | **1.95 GHz** |

- **The GPU changes its clock with the load.** One busy SM runs at ~2.9 GHz; all 36 at ~1.95 GHz (most likely the laptop's power or temperature limit).
- **In cycles, 36 SMs are 30.7× faster than 1; in seconds, 20.8×.** The ratio, 1.48, is the clock ratio. A strong-scaling curve in seconds mixes two effects: how well the work parallelizes, and how the hardware throttles. With `CYCLES` next to the time, they separate. This is why Hoefler and Belli ask for the clocks with every result ([hpc_architecture.md §4](hpc_architecture.md#4-benchmarking-like-hpc)).
- **At 36 SMs and N = 10⁸ the kernel lasts 15 ms,** short enough for the start and the tail (the last chunks) to matter too; the sweeps use a calibrated N (5 s at 36 SMs).

### One SM vs one CPU core (N = 10⁸)

| | one SM | one CPU core | one SM / one core |
|---|---|---|---|
| FP64 | 1.18·10⁷ photons/s | 0.95·10⁷ | **1.25** |
| FP32 | 3.2·10⁸ photons/s | 1.3·10⁷ | **~25** |

On this laptop, one SM is worth about one CPU core in FP64 (its FP64 units run at 1:64) and ~25 cores in FP32. The GPU wins by having 36 of them.

---

## Piece 20: memory, passes and the kernel's resources (`gpu_device.cu`, `gpu.cu`, `options.cu`, `report.cu`)

### What the processor now says about itself

`ProcessorInfo` (in `run.cuh`) gained two parts, both filled by `describe_processor` and printed in `CONFIG`, `SUMMARY` and every record (one list, `write_processor`, written with spaces on a line or one key per line in a record):

| Part | Keys | From (GPU) |
|---|---|---|
| `ProcessorHardware` | `MEMORY_TOTAL`, `MEMORY_FREE`, `MEMORY_FOR_IMAGE` (free − 256 MB of headroom), `L2_BYTES`, `PEAK_BANDWIDTH`, `FP32_PER_FP64`, `MAX_THREADS_PER_UNIT`, `CUDA_RUNTIME`, `CUDA_DRIVER` | `cudaMemGetInfo`, device attributes, `cudaRuntimeGetVersion`, `cudaDriverGetVersion` |
| `KernelResources` | `REGISTERS_PER_THREAD`, `LOCAL_BYTES_PER_THREAD`, `CONSTANT_BYTES`, `THREADS_PER_BLOCK`, `BLOCKS_PER_UNIT`, `BLOCKS`, `OCCUPANCY` | `cudaFuncGetAttributes(trace_photons)` and the launch from the occupancy API |

On the CPU: its RAM (`sysconf`), its 20 cores as `COMPUTE_UNITS_TOTAL`, and zeros where a CPU has nothing to say.

### Measured on the laptop

| | FP64 | FP32 |
|---|---|---|
| registers per thread | 80 | 67 |
| local memory (stack) per thread | 40 B | 32 B |
| constant memory (the Scene) | 18 048 B of 64 KB | 17 256 B |
| threads per block (the occupancy API's choice) | 768 | 896 |
| blocks per SM | **1** | **1** |
| **occupancy** = resident threads / 1536 | **0.50** | **0.58** |

GPU: 8.5 GB, 7.34 GB free (7.07 GB for the image), L2 32 MB, peak bandwidth ≈ 380 GB/s (2 × memory clock × bus width, as NVIDIA's `deviceQuery` computes it), `FP32_PER_FP64` = 64, CUDA runtime 13.3, driver up to 13.5.

### Registers limit the occupancy

- An SM has 65 536 registers. At 80 per thread there is room for 819 threads; in whole warps, 768. **Half of the SM's 1536 thread slots stay empty.** In FP32 (67 registers), 896: 58%.
- More resident threads hide memory latency better (when one warp waits, another runs); fewer registers per thread means values spill to local memory. **Which wins is an experiment, not a rule:** step 6 caps the registers (`-maxrregcount`, `__launch_bounds__`) and measures.
- EP3's kernel, 58 registers, was smaller; the star's physics is much bigger.

### Passes chosen from the GPU's free memory

- **With neither `--observers-per-pass` nor `--memory-budget`,** `main.cu` calls `fit_observers_per_pass(&options, memory_for_image)`: as many observers per pass as fit in the GPU's free memory, balanced (the CPU reports 0, so nothing changes there).
- **Measured:** 1k, 300 frames, 8 observers = 10 GB. The engine chose **4 + 4 in 2 passes** by itself (5 would fit; balanced to 4 + 4); 16 files written.

| Phase (both passes) | Time | |
|---|---|---|
| SETUP | 0.83 s | `cudaMalloc` of 5 GB + its first `cudaMemset`, the pinned buffer of 1.26 GB |
| PHOTONS | 0.12 s | N = 10⁷ only |
| COPY | 0.43 s | 8 × 1.26 GB, **~24 GB/s** device → pinned host |
| SAVE | **20.2 s** | 10 GB with `fsync`, ~0.5 GB/s |

- **At this size the disk is the serial part:** 20 s of saving against 0.12 s of photons. Amdahl's serial fraction grows with the image ([scaling_laws.md §3](scaling_laws.md#3-why-gustafson-matters-for-us)); for the showcase, N must be large enough for the photons to dominate.
- **Checked 2026-11-22 (Q3): not the idle clock; not reproduced.** Pass 2's kernel took 0.100 s here, pass 1's 0.021 s. Re-run with the clock sampled every 50 ms (1k, 300 frames, 8 and 12 observers, 4 per pass, N = 10⁷):
  - the clock does fall to ~520 MHz a few seconds into the saves, but a pass that starts after the drop (pass 3, at 11 s) was the fastest of the three (26.7 ms vs 30.2 and 29.8): the clock comes back within one sample;
  - pass 2 vs pass 1 over 12 runs at N = 10⁷ (kernels of ~30 ms): 0.71 to 1.37, in cycles too, either order of the observers; at N = 10⁸ (~170 ms): equal within 1%;
  - so the 5× was one sample of a 20 ms kernel, where run-to-run noise is ±30%. **Kernels shorter than ~0.1 s are not timed reliably**; the sweeps calibrated N for seconds.

### Checks

Passes and split on the GPU (FP64 and FP32), passes on the CPU, GPU = CPU at 10⁸: all **bit for bit**.

---

## Piece 21: the code mapped to the hardware, measured (`instruction_mix.py`, `kernel_experiments.py`, `gpu_monitor.py`)

The map of [hpc_architecture.md §7](hpc_architecture.md#7-the-code-mapped-to-where-it-runs-the-gpu-one-photon) made predictions; this piece measures them, without Nsight.

### The tools

| Tool | Does |
|---|---|
| **Build variants** (`engine.py`, `GPU_VARIANTS`) | the same sources with one flag, as their own binary: `sections` (`-DSTAR_SECTION_TIMERS`), `scene_in_global` (the Scene in global instead of constant memory), `registers_64`, `registers_40` (`-maxrregcount`), `no_fma` (`--fmad=false`). Never used for the normal runs |
| **Laps** (`photon.cuh`, `gpu_sections.cuh`) | `trace_photon(scene, rng, laps)`: `laps->lap(SECTION_…)` between its five pieces. The normal build passes `NoLaps`/`SilentLaps`, empty functions the compiler removes; the `sections` build passes `ClockLaps`, which reads `clock64()` and adds the cycles per section. One physics, instrumented only on request |
| **Kernel time** (`run_kernel`) | CUDA events around the launch: the GPU's own `KERNEL_TIME`, in a `KERNEL` line with `SCENE_MEMORY` |
| **`gpu_monitor.py`** | `nvidia-smi` sampled every 200 ms during a run: SM and memory clocks, power, temperature, utilization, memory used (min/median/max) |
| **`instruction_mix.py`** | `cuobjdump -sass` of `trace_photons`, each instruction counted by the hardware unit it uses |
| **`kernel_experiments.py`** | every variant × FP64/FP32, 3 repeats, with the monitor; compares the images with the baseline |

### The instruction mix (static: instructions in the code, not how often they run)

| Unit | FP64 kernel | FP32 kernel |
|---|---|---|
| FP64 units | 278 (152 `DFMA`, 68 `DMUL`) | 4 |
| FP32 cores | 73 | 247 (97 `FFMA`, 55 `FMUL`) |
| special function unit (`MUFU`) | 13 | 12 |
| integer units | **660** | **624** |
| atomics (L2) | 2 | 2 |
| constant loads; Scene operands (`c[0x3]`) | 46; 16 | 49; 18 |
| local memory (stack) | 5 | 20 |
| control flow (incl. `NOP`s) | 1261 | 296 |
| total | 2744 | 1520 |

- **Integer work is the largest class in both:** Philox (multiply-high, XOR: `IMAD`, `LOP3`) and the index arithmetic. It does not get cheaper in FP32.
- **Two atomic sites:** the image (`count_photon_atomic`) and the queue; the tallies' and the SM counters' atomics run once per thread.
- **The Scene is read straight from the constant bank** as instruction operands (`c[0x3]`): 16–18 places.
- **The 4 FP64 instructions in the FP32 kernel (Q1, checked 2026-11-22)** are inside CUDA's `sinf` and `cosf`, at the four calls in `photon.cuh` (lines 36–37 and 56–57, through `real_sin`/`real_cos`). Each is the same triple, `I2F.F64.S64` → `DMUL` by π/2·2⁻⁶⁴ → `F2F.F32.F64`: the **slow path of the argument reduction** (Payne–Hanek), which reads a table of the bits of 2/π from local memory, hence part of the FP32 kernel's 20 local-memory instructions. It runs only for |x| above ~10⁵; our angles stay within [0, 2π], so it **never runs**. Found with `-Xptxas --warn-on-double-precision-use` and the PTX's `.loc` lines. *(An idea, not done: `sincospif(2u)` for an azimuth 2πu needs no reduction at all; it would change the FP32 image's last bits.)*
- **The 1063 `NOP`s of the FP64 kernel (Q2, checked 2026-11-22) are not padding: they are waits.** They come in runs of 4, between FP64 instructions that depend on each other (e.g. `MUFU.RCP64H` → `DFMA` → `DFMA` …, the Newton steps of a division): 233 of the 303 runs have 4 `NOP`s, and the FP32 kernel's runs of 4 sit exactly around its 4 FP64 instructions. On a 1:64 GPU an FP64 result arrives late, and the compiler makes the warp wait for it with `NOP`s. They are executed; while one warp waits, the SM runs another, which is one reason FP64 is 17× slower here and not 64× (my first guess, alignment padding, was wrong).

### The experiments (the whole GPU, N = 5·10⁸ FP64 / 5·10⁹ FP32, 3 repeats, spread < 1%)

| Variant | registers | spills | occupancy | FP64 vs baseline | registers | spills | occupancy | FP32 vs baseline |
|---|---|---|---|---|---|---|---|---|
| baseline | 80 | 0 | 0.50 | 1 (6.46·10⁸ photons/s) | 67 | 0 | 0.58 | 1 (1.11·10¹⁰ photons/s) |
| Scene in global memory | 88 | 0 | 0.42 | 1.002 | 68 | 0 | 0.58 | 0.984 |
| registers ≤ 64 | 64 | 24 B | 0.67 | 0.989 | 55 | 0 | 0.75 | 0.965 |
| registers ≤ 40 | 40 | 344 B | 1.00 | **0.911** | 40 | 80 B | 1.00 | 0.952 |
| no FMA | 80 | 0 | 0.50 | **0.927** | 67 | 0 | 0.58 | 0.980 |
| section timers | 112 | 0 | 0.33 | 1.006 | 69 | 0 | 0.58 | **0.922** |

- **Constant memory is not the bottleneck:** the Scene in global memory runs at the same speed. The `slot_of_imu` suspect (a different index per thread) is cleared: one table read per photon, against hundreds of arithmetic instructions.
- **More occupancy does not help; spills hurt.** Capping the registers raises occupancy to 0.67–1.0 and is slower in both precisions. The occupancy API's choice is the right one for this kernel: there is enough arithmetic per thread to keep the SM busy with half its thread slots.
- **FMA matters in FP64:** 7% faster with it. In FP64 the images are identical with and without it (0 pixels); in FP32, 30 689 pixels differ at N = 5·10⁹ (as in piece 18).
- **The observer effect:** the timers cost nothing in FP64 (even with 112 registers) and 8% in FP32.
- **All images equal the baseline,** except `no_fma` in FP32: capping registers, moving the Scene, timing sections change speed, not results.

### Where the cycles go (the `sections` variant)

| Section | FP64 | FP32 | What it does |
|---|---|---|---|
| direction | 33% | **41%** | μ (`sqrt` or `cbrt`), the local azimuth (`cos`, `sin`) |
| surface | 20% | 28% | z, φ, `cos`, `sin`, `sine_from_cosine` |
| bin | **22%** | 8% | `atan2` (expensive in FP64), the slot table |
| record | 15% | 13% | the tally, and the image's atomic for the 8% recorded |
| spots | 8% | 9% | 3 chord tests, sometimes a random number |
| pixel | 1.4% | 0.9% | the projection, for the 8% that reach an observer |
| chunk | 0.0% | 0.0% | taking a chunk, `curand_init` (once per 1024 photons) |

(Cycles as each thread sees them, stalls included; the shares compare pieces of the same kernel.)

### What this says about the kernel

1. **It is limited by arithmetic** (roots, trigonometry), not by memory: constant vs global makes no difference, and the image's atomics are 13–15% of the time.
2. **FP64 is 17× slower than FP32 per photon, not 64×,** on a GPU whose FP64 units run at 1:64: a large part of the work is integer and control (Philox, indices, branches), which costs the same in both. Amdahl's law inside the kernel: only the FP64 fraction slows down.
3. **The clock depends on the work:** the FP32 kernel ran at a median 2310 MHz drawing ~72 W, the FP64 one at 2812 MHz drawing ~55 W. FP32 keeps more units busy, so it hits the power limit sooner (medians of a few 200 ms samples: indicative).
   - **Not the spills (checked 2026-11-22):** the baseline has no spills and still drops its clock; across the variants the clock follows the power, not the spill bytes (FP32 with 80 B of spills ran at 2467 MHz, above the baseline's 2310).
   - **The driver says why** (`nvidia-smi`'s `clocks_event_reasons`, sampled under load): FP64 on 36 SMs and FP32 on 1 SM sit at 2880–2895 MHz at 36–56 W with only `reliability` active, the top of the voltage–frequency curve. FP32 on 36 SMs reaches the power limit: `sw_power_cap` active, and the enforced limit itself moved between 85 and 115 W within one run (the laptop sharing its power between CPU and GPU; likely NVIDIA's Dynamic Boost, not confirmed). That run reached only 7.4·10⁹ photons/s, against 9–11·10⁹ on other days.
   - **Why FP64 draws less:** its warps spend much of their time waiting for the two FP64 units per SM (the `NOP`s above), so fewer transistors switch per cycle.
4. **Closed 2026-11-22:** pass 2 of piece 20 running 5× slower was noise on a 20 ms kernel, not the idle clock (piece 20, "Checked").

---

## Piece 22: the sweeps (`sweeps.py`)

The definition is in [gpu_engine.md §7](gpu_engine.md#7-the-sweeps-today-to-review); this is the code that runs it.

### One point = one engine process

```python
def run_point(point, scratch_folder):          # point = {processor, precision, sms, photons, repeat}
    config = engine.default_star_config(point["photons"], sms=point["sms"])
    with GpuMonitor() as monitor:              # nvidia-smi every 200 ms, next to the run
        output = engine.run_engine_output(config, scratch_folder, processor, precision)
    → SUMMARY, KERNEL, SMS lines + the monitor's clocks + Python's PROCESS_TIME; the image is deleted
```

- **The default star** (EP2's: 200², 150 frames, 4 observers, one pass, recorded) comes from `engine.default_star_config`, the same one `kernel_experiments.py` uses.
- **Each point is saved as its own file** (`point_gpu32_sms09_n5.50e+10_rep2.json`), written `.partial` then renamed: a file that exists is a finished point.

### A sweep = a list of points, decided first

```python
describe_sweep(kind, precision, seconds, repeats)   → sweep.json: what was asked, the code hash, every point
run_sweep(folder)                                   → runs only the points without a file
```

| Kind | Points | N |
|---|---|---|
| `strong` | SMs 1, 2, 3, 4, 6, 9, 12, 18, 24, 36, the same N | calibrated: 5 s of photons on 36 SMs |
| `weak` | the same SMs, N ∝ SMs | calibrated: 5 s per point |
| `problem-size` | N = 10⁵ … 10¹⁰, each on 1 and on 36 SMs (FP32) | fixed list |
| `devices` | `cpu64`, `cpu32`, `gpu64`, `gpu32`, all units | 10⁸ |

- **Calibration** (`speed_on_all_units`): one short run on all SMs measures the photons per second; N = seconds × speed. The resulting N is written in `sweep.json`, so the sweep can be repeated exactly.
- **Interleaved repeats:** repeat 1 of every point, then 2, then 3, so a slow drift (the laptop heating) spreads over all points instead of biasing the last ones.
- **A warm-up run** before the first point, not saved: the GPU starts from its idle clock and the first CUDA call loads the driver.
- **Resuming:** `python sweeps.py continue <folder>` runs only the missing points (tested: 2 deleted points were the only ones rerun).
- **`python sweeps.py all`** runs the six sweeps in turn: devices, problem-size (FP32), weak (FP64, FP32), strong (FP64, FP32).

### While it runs

Nothing else should use the GPU: another program would change the clocks and the timings. The full set takes about 50 minutes (the 1-SM points of `strong` are the longest: ~3–5 min each).

---

## Piece 23: the scaling analysis (`scaling.py`)

Reads the latest sweep of each kind, takes the **median of the repeats** for each point, fits the laws, and saves each figure as a `.png` with its data as `.json` (so the slides can be redrawn without rerunning), plus a `report.md`.

### The laws, as code

```python
def fit_amdahl(sms, times):          # T(n) = T(1) · (s + p/n): linear in 1/n, least squares
    columns = [1, 1/n]  →  serial, parallel  →  s = serial / T(1), p = parallel / T(1), ceiling = 1/s

def fit_gustafson(sms, scaled):      # S(n) = n − s′(n − 1): one parameter, least squares
    s′ = Σ (n − S)(n − 1) / Σ (n − 1)²
                                     # S(n) measured = (serial + n · photons(1)) / T(n): one SM's time for the
                                     # n-fold problem, over n SMs' time (Gustafson's definition)

def parallel_fraction(speedup):      # p from two points (1 and 36 SMs): Amdahl solved for p
    p = (1 − 1/S) / (1 − 1/36)
```

### The figures

| Figure | From | Shows |
|---|---|---|
| `amdahl_ceiling_fp32` | `strong` at N = 10⁸, `strong` at large N, `weak` | the same GPU, three questions: Amdahl's flattening (small N), near-ideal strong scaling (large N), Gustafson's line |
| `strong_fp64`, `strong_fp32` | `strong` | speedup vs SMs for the **photon phase**, the **whole run**, and the photon phase **in cycles** (the clock removed); the ideal line; the Amdahl fits (dashed). A second panel: the SM clock per point (one axis per panel, never two on one) |
| `weak_fp64`, `weak_fp32` | `weak` | time per point (ideal: flat) and the scaled speedup, (serial + n · photons(1)) / T(n), with the Gustafson fit |
| `strong_vs_weak_fp64/32` | both | the flattening curve and the straight line on one plot: the class's central figure |
| `problem_size` | `problem-size` | p vs N, photon phase and whole run |
| `devices` | `devices` | photons per second, the four binaries (log scale) |

- **Three speedups, three questions:** the photon phase (how well the kernel parallelizes), the whole run (what the user waits for, with the context creation), the cycles (the same, without the clock changing under it, piece 19).
- **The serial part, two ways:** fitted (Amdahl's s) and measured from the phases ((setup + zero + copy + save) / T(1)); they should agree.
- **Speedups always at the same precision:** GPU FP32 against CPU FP32, GPU FP64 against CPU FP64; the CPU is one core, said in every title.

### First results (the finished sweeps)

**p(N), FP32 (medians of 3):**

| N | p, photon phase | p, whole run |
|---|---|---|
| 10⁵ | 0.14 | ≈ 0 |
| 10⁶ | 0.49 | ≈ 0 |
| 10⁷ | 0.92 | 0.07 |
| 10⁸ | 0.989 | 0.54 |
| 10⁹ | **0.9993** | 0.92 |
| 10¹⁰ | 0.997 | **0.987** |

- **EP3's result, for the star:** p is not a property of the program; it grows with N. For the whole run it stays near 0 until the photons outweigh the context creation (~0.15 s).
- p slightly *below* 1 at 10¹⁰ in the photon phase: the 36-SM run lasts long enough for the clock to drop, while the 1-SM run keeps its boost (piece 19). The cycle-based speedup separates this in `strong`.
- p ≈ 0 (even −0.02) for the whole run at 10⁶: both runs are ~0.3 s of setup; the difference is noise.

**Devices (N = 10⁸, the GPU on all 36 SMs, the CPU on one core):**

| | photons/s | whole run | GPU vs CPU, same precision: photon phase | whole run |
|---|---|---|---|---|
| CPU FP64 | 7.1·10⁶ | 14.2 s | | |
| CPU FP32 | 1.1·10⁷ | 9.6 s | | |
| GPU FP64 | 5.9·10⁸ | 0.52 s | **82×** | **28×** |
| GPU FP32 | 9.0·10⁹ | 0.33 s | **850×** | **29×** |

- **The CPU ran slower here (7.0–8.0·10⁶, the three repeats) than on other days (8.8–9.5·10⁶). Checked 2026-11-22 (Q4): the cause is not found, and my first explanation (the shared power budget) is not supported.** N = 10⁸ FP64 on the CPU gave 8.8–9.1·10⁶ under every condition tried: pinned to each of the 20 vCPUs (`taskset`; WSL does not show which are P- or E-cores, and Windows moves the vCPUs anyway), for 11 s, right after 37 s of the GPU at full load, and with `nvidia-smi` polling every 200 ms. Nothing else was running during the sweep (it was the first). What WSL cannot see (Windows' power mode, AC or battery, other Windows programs) is left. **So the one-core baseline varies by ~25% between sessions,** and the speedups with it: against today's CPU, FP64's 82× would be ~65×.
- **The whole-run speedup is ~28× in both precisions:** at N = 10⁸, the GPU's run is mostly setup, which does not depend on the precision.

### The full results (strong and weak; medians of 3)

**Strong scaling (N calibrated for 5 s on 36 SMs):**

| | FP64, N = 3.1·10⁹ | FP32, N = 5.7·10¹⁰ |
|---|---|---|
| speedup on 36 SMs: photon phase | **35.9** | 33.1 |
| … in cycles (the clock removed) | 36.0 | **36.0** |
| … whole run | 33.8 | 31.5 |
| Amdahl fit, photon phase | p = 0.99980 | p = 0.99916 |
| Amdahl fit, whole run | p = 0.99782 (s = 0.0022, ceiling 458) | p = 0.99767 (s = 0.0023, ceiling 429) |
| serial part measured from the phases | 0.0019 | 0.0015 |
| SM clock, 1 → 36 SMs | 2887 → 2880 MHz | 2895 → **2636 MHz** |

- **FP64 scales perfectly over the 36 SMs** (35.9× of 36), and the clock does not move: the FP64 kernel stays under the laptop's power limit.
- **FP32 scales perfectly in cycles (36.0×) but 33.1× in seconds:** on 24 and 36 SMs the clock falls (2850, 2636 MHz). In FP32, Amdahl's fitted "serial part" is mostly the hardware throttling, not serial code.
- **The serial part, two ways, agree** for the whole run: fitted s = 0.0022 vs measured 0.0019 (FP64).

**Weak scaling (5 s per point, N ∝ SMs):**

| | FP64 | FP32 |
|---|---|---|
| photon phase, 1 → 36 SMs | 4.67 → 4.59 s (flat) | 4.90 → **5.31 s** |
| efficiency on 36 SMs | 1.02 | **0.92** |
| scaled speedup on 36 SMs (whole run) | 34.1 | 31.8 |
| Gustafson fit s′ (whole run) | **0.054** | 0.089 |
| serial share measured on 36 SMs | **0.058** | 0.044 |

- **FP64: Gustafson's law and the measurement agree** (s′ fitted 0.054, measured 0.058): the constant serial part (~0.3 s of setup) is all there is.
- **FP32: the fit's s′ is twice the measured serial share,** because the photon phase itself grows by 8% on 36 SMs: the clock again.

**A mistake of mine, corrected:** the first version computed the weak "scaled speedup" as n · T(1, W) / T(n, nW), which counts the serial part n times; its Gustafson fit came out *negative* (s′ = −0.013), meaningless. Gustafson's definition is the time *one* SM would take for the *n-fold* problem (the serial part once, plus n times the parallel part), over the time n SMs take: **S(n) = (serial + n · photons(1)) / T(n)**. With it, the fit and the measurement agree (FP64). The photon-phase line, n · T(1, W)/T(n, nW), is kept as "n × efficiency".

### What the class can take from it

1. **Strong scaling at large N is almost ideal on this GPU (p = 0.998–0.9998):** Monte Carlo is embarrassingly parallel, and the queue balances the SMs.
2. **But p depends on N** (the `problem-size` table): at 10⁸ photons the whole run is half setup (p = 0.54).
3. **The clock is part of the result:** the same code in FP32 throttles the laptop, in FP64 it does not. Measured in cycles, both scale perfectly; measured in seconds, FP32 loses ~8% on all 36 SMs.
4. **Strong at large N looks like weak:** with p ≈ 0.998 both curves are nearly straight. Amdahl's flattening needs a small N: the small-N strong sweep (N = 10⁸) shows it (`amdahl_ceiling_fp32`).

### Amdahl's ceiling, measured (strong, FP32, N = 10⁸)

| SMs | photon phase | whole run | speedup, photons | speedup, whole run |
|---|---|---|---|---|
| 1 | 0.329 s | 0.580 s | 1 | 1 |
| 6 | 0.057 s | 0.294 s | 5.8 | 2.0 |
| 36 | 0.011 s | 0.251 s | 29.7 | **2.3** |

- **Whole run: p = 0.58, ceiling 1/s = 2.4.** The serial part fitted (s = 0.42) and measured from the phases (0.41) agree. Thirty-six SMs make the run 2.3× faster: the ~0.25 s of setup (the CUDA context) is the wall.
- **The photon phase alone still scales** (p = 0.994): the kernel is as parallel as ever; the program around it is not.
- **The figure `amdahl_ceiling_fp32`:** the same GPU, three questions. Small N strong (flat at 2.3×), large N strong (p = 0.998, nearly straight), weak (Gustafson's line). The class's central plot.

---

## Piece 24: the showcase (`showcase.py`)

```
python showcase.py --size 1024 --frames 300 --snr 10 --precision 32
```

1. **N from the SNR wanted at the limb:** N = SNR² · nmu · frames · (π/4) · size² · (1 + 2a/3) ([jobs_and_passes.md §10](jobs_and_passes.md#10-photons-per-pixel-snr-and-n-always-both-ways)); `nmu` from the engine's `--describe`.
2. **The run** on the GPU, FP32, with `nvidia-smi` sampled.
3. **Per observer:** a still (frame 0) and a 10 s mp4 at 30 fps (`ffmpeg`, H.264), each frame read from the file on disk with `np.memmap`: a 1.26 GB observer is never loaded whole.
4. **Measured vs predicted SNR** in two rings: the disk's center (r < 0.2) and near the limb (0.90 ≤ r < 0.95).

### The run

| | |
|---|---|
| image | 1024², 300 frames, nmu 95, 4 observers in **1 pass** (5.03 GB on the GPU) |
| photons | **N = 3.9·10¹²** (SNR 10 at the limb) |
| photon phase | **385 s at 1.0·10¹⁰ photons/s** (whole run 393 s) |
| on one CPU core | ~4.3 days at the CPU's 1.06·10⁷ photons/s (FP32) |

### The SNR, measured vs predicted

| observer | center: measured / predicted | limb ring: measured / predicted |
|---|---|---|
| i = 1.5708 (equator-on) | 14.1 / 14.1 | 11.7 / 11.7 |
| i = 1.0411 | 14.1 / 14.1 | 11.7 / 11.7 |
| i = 0.5293 | 14.0 / 14.1 | 11.7 / 11.7 |
| i = 0.1452 (pole-on) | 13.6 / 14.1 | 11.7 / 11.7 |

- **The planning formula and the run agree** to the first decimal: the photons land where the physics says, in the numbers it says.
- The limb ring (mean μ ≈ 0.38) predicts 11.7; the formula's SNR 10 is for μ → 0, the very edge.
- The pole-on center is slightly lower (13.6): most likely a spot near the pole passing through the central ring in some frames.

### The files

- `observer_…_frame_0.png` (1.3 MB each) and `observer_….mp4` (~49 MB each, 10 s): in `runs/…-showcase-1024-f300-snr10-gpu32/`. The videos stay out of git (`*.mp4` in `.gitignore`), as the 5 GB of raw images do (`*.bin`); the stills, the records and `showcase.json` are committed.

---

## Piece 25: the products (`main.py`, `lab/`)

By piece 24 the engine worked and was measured, but using it meant knowing seven scripts, each with its own N and its own folder layout, and nothing said whether a run was a valid experiment. Piece 25 rebuilds the outside around **what you want to do**:

```
python main.py test        does it work here?
python main.py make        pictures and videos
python main.py measure     every measurement on this machine, at valid N, with its report
python main.py compare     two machines side by side
```

### One shape for every product

1. **Check the environment** (`lab/environment.py`): the compiler, the GPU, other processes on the GPU, disk, the packages, uncommitted engine code, whether this host's speeds have been measured. A failure stops; a warning is shown.
2. **Plan and predict** (`lab/plan.py`): every engine run it will make, with its predicted seconds (N over the speed measured here, plus the serial part), the passes, the device memory and the disk; and whether each run is **valid**.
3. **Ask** (`--plan` stops here, `--yes` doesn't ask).
4. **Run, then analyse, then write** the run's folder.

### Valid runs: the minimum N

A speed is a measurement only if the GPU was busy long enough and evenly enough:

N_min = max(1 s × the speed, 32 chunks × the resident threads × 1024 photons per chunk)

- **1 s:** below it, the launch and the clock's ramp-up dominate.
- **32 chunks per resident thread:** the work queue hands out whole chunks; with too few, some SMs get one more chunk than others and the slowest decides the time. At 10⁷ on the laptop there are 9 766 chunks for 27 648 threads, and the same pass took 0.6–1.6× as long from one run to the next ([research.md, Q3](research.md)).

On the laptop, on all SMs: FP64 9·10⁸ (set by the work), FP32 1.1·10¹⁰ (set by time). The A30 will compute its own.

### The unit of speed is the SM, not the CPU

A whole GPU against one core of a sequential CPU program is the classic unfair comparison: it credits the GPU with every core the CPU didn't use and every optimization the CPU code never got (Lee et al., *Debunking the 100X GPU vs. CPU myth*, ISCA 2010). Our first report did exactly that ("70× one core"). So the products measure the GPU against **itself**: one SM, then all of them (35.2× FP64, 32.9× FP32 on 36 SMs, 35.9× in cycles). The sequential CPU engine stays only as the correctness reference. Comparing EP4 with EP2 is a different question, left for the TCD.

### No small N

Every timed run in `measure` is above the minimum N. The small-N strong sweep (N = 10⁸) and the bottom of p(N) (10⁵…10⁹) were there "below the minimum by design" to show Amdahl's ceiling; they are gone (2026-11-22). At 1k they said nothing (36 SMs: 1.07×), and the ceiling shows at valid N anyway: on 36 SMs a run is 6.0 s of photons and 5.6 s of setup, copying back and saving the 5 GB image, so the whole run is 19× (FP64) where the photons are 36×, and its ceiling is 1/s ≈ 40. The serial part of a high-resolution picture is the image. p(N) now runs at 0.4×, 1× and 2× the reference N (FP32), all valid.

### `measure`: the runs are shared, the report explains them

- **Parts:** `validation` (GPU = CPU bit for bit; the GPU vs the sequential CPU with independent photons, χ²; FP32 vs FP64; and image digests at a fixed N), `speed` (one SM and all SMs: the SM is the unit, the CPU's speed isn't compared), `kernel` (resources, instruction mix, variants), `occupancy` (registers → warps → occupancy, the register caps interleaved and compared in cycles, the A30 compiled), `scaling` (strong and weak, FP64 and FP32, Amdahl's ceiling and p(N), all at valid N), `costs` (no runs: resolution vs image size, what each costs in photons, time, memory and passes, and the blur the bins leave; from the measured speed and the GPU's memory).
- **A run is a point** (`lab/points.py`): one engine process, one `points/<binary>-<variant>-<sms>-<N>-<repeat>.json`. Parts that need the same run share it: the reference run on all SMs is at once the speed row, the kernel's baseline, the uncapped register run and the strong sweep's last point. `--continue` runs only the missing points.
- **The report** (`lab/explainer.py`, `lab/report.py`): a notebook built from the data and executed (nbclient), then a PDF (nbconvert → LaTeX → tectonic). Each section has the question, the method, the raw table (read from `parts/*.json` when the notebook runs), the figure, what it means with the numbers, the caveats and where the data is. An appendix describes every field. The notebook is meant to grow into the scientific report.

### The map from the old names

| Then | Now |
|---|---|
| `engine/*.cu*` (one folder) | `engine/physics/`, `random/`, `run/`, `cpu/`, `gpu/`; `main.cu` and `processor.cuh` at the top |
| `engine.py`, `analysis.py`, `gpu_monitor.py`, `instruction_mix.py`, `scaling.py`, `showcase.py` | the same modules in `lab/` |
| `star.py -n …` (a picture) | `main.py make -n …` (or `--snr`) |
| `star.py --check-split`, `--check-passes`, `--compare-gpu` | `main.py test` (at 10⁷), and `measure`'s validation part (at valid N) |
| `star.py --compare-ep2` (EP4 vs EP2's own program, χ², no spots) | **removed (2026-11-22):** the GPU is compared with EP4's sequential CPU engine instead, with independent photons of the same star (the same seed, different chunks), χ²; spots stay on |
| `star.py --compare-precision` | `measure`'s validation part |
| `instruction_mix.py`, `kernel_experiments.py` | `measure --parts kernel` (`kernel_experiments.py` is gone; its variants are points) |
| the register caps (scratch) | `measure --parts occupancy`, and `research/register_sweep.py` |
| `sweeps.py all` + `scaling.py` | `measure --parts scaling` (`sweeps.py` is gone; the sweeps are points) |
| `showcase.py --size 1024 --frames 300 --snr 10` | `main.py make --size 1024 --frames 300 --snr 10` |
| a gif per observer (or an mp4 when `ffmpeg` was there) | **mp4 only**: `make` needs `ffmpeg` and stops if it's missing (2026-11-22) |
| `showcase.py` (N from the SNR, the videos, the SNR check) | `lab/snr.py` (N from the SNR, the SNR check) and `lab/videos.py` (the videos) |
| one 10 s video per observer, playing one rotation once | per observer, plus `observers_grid.mp4` (all side by side) and `observers_around.mp4` (each view placed around a drawing of the star at its inclination from the rotation axis, pole-on at the top); every video repeats the rotation `--loops` times (default 5) |

### The videos

- **One rotation is exactly `--frames` frames.** The engine bins each photon's azimuth over the full 2π into `--frames` frames, so the last frame is one step before the first (checked on the 1k showcase: frame 299 differs from frame 0 as much as two neighbouring frames do, 12.5 vs 12.4, against 14–16 half a turn away). At 30 fps, 300 frames are 10 s: short, and the player stops at the end, which made it look like less than a rotation.
- **Repeating without re-encoding:** each video is encoded once (one rotation), then `ffmpeg -stream_loop` copies it `--loops` times. The size grows with the loops (~0.15 bytes per pixel per frame); `make` predicts it before running.
- **Around the star:** the views sit on a circle around a drawing of the star and its rotation axis, at their inclination (the angle from the axis). The circle is fixed; when two observers are close in inclination, the views shrink so that they don't overlap.
- **Both hemispheres:** `make`'s default is 7 observers, i = 0°, 30°, 60°, 90°, 120°, 150°, 180°: the three from the pole to the equator, mirrored through the equator. In the around video they make a half circle from the north pole (top) to the south pole (bottom). `measure` keeps its 4 observers, so measurements stay comparable. 7 observers of 1024² × 300 frames need 8.8 GB, so they run in 2 passes, and each pass traces all N photons again: the time doubles (the plan's prediction now counts the passes).
| `engine.py --arch 80` | `main.py build --arch 80` |
| `runs/<stamp>-<kind>/` | `runs/<product>/<stamp>-<label>/`, with `run.json`, `summary.md`, and `runs/index.md` |
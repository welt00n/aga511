# EP4 walkthrough

A guided tour of the EP4 sequential engine, one piece at a time, with the questions that came up on the way. Code: [`../EP4/engine/`](../EP4/engine/).

| # | Piece | File(s) | Status |
|---|---|---|---|
| 1 | One run from start to finish | `main.cu` | done |
| 2 | The `real` type and `HD` | `real.cuh` | done |
| 3 | Random numbers: Philox | `rng.cuh` | done (the *how* of the rounds: to revisit); C++ syntax and headers in [cpp_for_python.md](cpp_for_python.md) |
| 4 | Chunks: dividing the work | `main.cu` | done |
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
    unsigned int low_0 = multiply_high_low(PHILOX_M4x32_0, counter.x, &high_0);
    unsigned int low_1 = multiply_high_low(PHILOX_M4x32_1, counter.z, &high_1);
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

- **Every detail matches the CUDA 13.3 headers:** the constants, the round, the counter layout, the quirky carry in `philox_skip_blocks`, the conversion to a uniform number. That is why `PHILOX_M4x32_0`, ... keep cuRAND's names.
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

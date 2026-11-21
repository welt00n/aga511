# Concepts notebook

Everything we learn while building EP3 and EP4, written down so it can become the CUDA class and the TCD monograph.

This file is the **index**: every topic gets a short description, what caught our interest, and where it shows up in the code. When a topic grows, it moves to its own file in this folder and this index links to it.

Status tags: **[decided]** a choice made for EP4 · **[measured]** checked on this machine · **[to explore]** not done yet.

**Where each topic is:**

| Topic | Where |
|---|---|
| **The plan: phases A–I, how we got here, the log of decisions and findings** | [roadmap.md](roadmap.md) |
| The EP4 engine, piece by piece, with the questions asked | [walkthrough.md](walkthrough.md) |
| Piece 1: one run, the phases, the cost of counters and logging | [walkthrough.md](walkthrough.md#piece-1-one-run-from-start-to-finish) |
| Piece 2: `real.cuh`, `HD`, overloading, `sine_from_cosine` | [walkthrough.md](walkthrough.md#piece-2-realcuh-the-precision-switch), [precision.md §8](precision.md#8-deep-dive-sine_from_cosine-and-the-fp32-correction) |
| Piece 3: Philox (counter, key, rounds, a stream per chunk, the cuRAND copy) | [walkthrough.md](walkthrough.md#piece-3-random-numbers-philox-rngcuh) |
| C++ for a Python programmer: the build, headers, `static inline`, include guards, templates with `HD`, the syntax | [cpp_for_python.md](cpp_for_python.md) |
| FP64/FP32/FP16, random bits, the sine deep dive | [precision.md](precision.md) |
| Amdahl (strong scaling) and Gustafson (weak scaling), and the EP4 scaling experiments | [scaling_laws.md](scaling_laws.md) |
| Jobs, observers and passes: fitting any image in any machine's memory, resuming over days | [jobs_and_passes.md](jobs_and_passes.md) |
| The driver: commands, job files and prompts, remote runs, the engine's records, sweeps and the Amdahl/Gustafson analysis | [driver.md](driver.md) |
| The laptop and the IAG A30, predictions, questions for IAG | [gpus.md](gpus.md) |
| Nsight: how it works, Buffon's profile, the permission error | [../cuda_class/nsight.md](../cuda_class/nsight.md) |
| The class plan | [../cuda_class/class_plan.md](../cuda_class/class_plan.md) |
| Everything else (memory, atomics, the queue, chunks, Amdahl, containers, tools, libraries) | the sections below |

**Documents in this folder:**

- [roadmap.md](roadmap.md): the one plan (phases A–I), the story so far, the log of decisions and findings.
- [walkthrough.md](walkthrough.md): the EP4 engine, piece by piece, with the questions asked along the way.
- [cpp_for_python.md](cpp_for_python.md): C/C++ for a Python programmer: the build, headers with whole functions, how EP4's files are organized, the new syntax.
- [precision.md](precision.md): FP64, FP32, FP16, random bits, and the deep dive on `sine_from_cosine`.
- [gpus.md](gpus.md): the laptop GPU and the IAG A30 side by side, predictions, questions for IAG.
- [scaling_laws.md](scaling_laws.md): Amdahl and Gustafson, and the EP4 scaling experiments.
- [jobs_and_passes.md](jobs_and_passes.md): jobs, observers, passes, idempotent runs, machine analysis.
- [driver.md](driver.md): the Python driver's commands, local and remote, and the scaling sweeps and analysis.
- [experiments/](experiments/) and [figures/](figures/): small scripts behind the measurements, and their plots.
- For the class: [../cuda_class/class_plan.md](../cuda_class/class_plan.md), [../cuda_class/nsight.md](../cuda_class/nsight.md).

---

## 1. The course and the deliverable

- **EP4 is the TCD** (Trabalho de Conclusão de Disciplina, 35% of the grade, due 01/12). The EP2 star rebuilt with CUDA, plus a monograph that extends the EP2 one with: a summary of the technique, how it was implemented, and performance tests (Amdahl's law, speedup, ...). Also a 10–15 min individual presentation. Source: `01 - introducao.pdf`.
- **The algorithm stays the professor's** (binned observers, from `02 - projeto.pdf`). **[decided]**

## 2. CUDA programming model

- **Host and device.** One `.cu` file holds two programs: CPU code and GPU code, both compiled by `nvcc`. The CPU is in charge: it allocates GPU memory, copies data, launches kernels and waits for them. EP3: `run_on_gpu`.
- **Function qualifiers.** `__global__` = a kernel (launched by the CPU, runs on the GPU); `__device__` = GPU-only helper; `__host__ __device__` = compiled for both, which is how one physics function will serve the CPU and the GPU paths.
- **Grid → blocks → threads → warps.** A kernel launch is a grid of blocks. Each block runs on one SM, its threads run in warps of 32 that execute the same instruction together (SIMT). RTX 5070 Laptop: 36 SMs, 128 FP32 cores per SM. **[measured]**
- **Occupancy.** How many blocks fit on an SM at the same time, limited by registers, shared memory and block size. EP3's `choose_launch` asks the occupancy API (`cudaOccupancyMaxPotentialBlockSize`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`). `__launch_bounds__` tells the compiler the block size to plan registers for.
- **Persistent threads.** Launch exactly enough blocks to fill the GPU once, and let the threads loop over the work, instead of one thread per needle or photon.
- **Cooperative launch.** Guarantees every block is resident at the same time. EP3 needs it so blocks that quit early (disallowed SMs) are not replaced by new ones.
- **Reading the SM id (`%smid`, inline PTX).** CUDA has no "use only n SMs" option. EP3 builds one: find the real SM ids, allow the first n, and blocks on other SMs return at once. This is what makes n = SMs in Amdahl's law.

## 3. Memory

- **Two address spaces.** The GPU has its own memory (VRAM). A pointer from `cudaMalloc` holds a GPU address: the CPU must never dereference it, and reaches that memory only through `cudaMemcpy`.
- **Lifetime outside kernels.** A `cudaMalloc` allocation lives from `cudaMalloc` to `cudaFree`, across any number of kernel launches. It is owned by host code. Kernel locals live in registers and die with the thread.
- **Kernel arguments are copied by value.** Each thread gets a copy of the *pointer*, all copies point at the same memory, and that is how threads share the queue and the image.
- **The memory spaces.** Registers (one thread), shared memory `__shared__` (one block, ~100 KB per SM), global memory `cudaMalloc` (everyone, plus the host via copies), constant memory `__constant__` (read-only in kernels, filled with `cudaMemcpyToSymbol`, 64 KB, broadcast-cached).
- **"Local memory" is not local.** A per-thread array indexed at run time is placed in local memory, which physically lives in global memory (cached). Registers are the only fast private storage.
- **Small globals without malloc.** A `__device__` variable set with `cudaMemcpyToSymbol`, handy for a single counter.
- **Managed and pinned memory.** `cudaMallocManaged` gives one pointer valid on both sides (convenient, hides the copies). `cudaMallocHost` (pinned RAM) makes big copies faster, relevant for copying a 1.5 GB image back. **[to explore]**
- **EP4 placement.** Queue and image in global memory, the `Scene` (spots, limb darkening, observer table) in constant memory. **[decided]**
- **A pointer does not say where it points.** An `unsigned int *` holding a GPU address looks exactly like one holding a RAM address; mixing them compiles and then crashes or reads garbage.
- **Recording: shared physics, separate recording per side.** **[decided]**
  - The CPU keeps `Image` + `count_photon` (`counts[pixel]++`).
  - The GPU gets its own type, `DeviceImage`, with `allocate_device_image` (`cudaMalloc` + `cudaMemset`), `copy_to_host`, and a `__device__ count_photon_atomic` (`atomicAdd`), where the faster variants (warp aggregation, shared memory) will go.
  - Rejected: one `HD` `count_photon` branching on `__CUDA_ARCH__`. It hides the atomic, lets a RAM image reach the kernel without a compile error, and would keep growing with GPU-only variants.
  - Separate types make the compiler reject the wrong image on the wrong side. It is the same split as the queue and the counters: the physics is shared; how work is taken and results are written belongs to each side.

## 4. Synchronization, atomics and the work queue

- **Races.** `x++` is read, add, write. Two workers can read the same value and both write the same result.
- **Atomic fetch-and-add.** Read and increment as one indivisible operation that returns the old value. The same idea in C11 (`atomic_fetch_add`), OpenMP (`#pragma omp atomic capture`, or `schedule(dynamic)`, which is a built-in queue) and CUDA (`atomicAdd`).
- **The work queue is one integer.** `next_chunk`, the next chunk nobody took. Sequential CPU: a local variable and `++`. CPU threads: a shared atomic in RAM. Containers: no shared memory, so the range is split up front (or a queue goes in an outside service). GPU: a counter in global memory and `atomicAdd`. EP3: `take_next_chunk`.
- **Where GPU atomics run.** In the L2 cache, in atomic units shared by all SMs. Atomics on the same address are serialized there.
- **Order does not matter for integer sums.** GPU chunks finish in any order. The image holds integer counts, and integer addition gives the same result in any order, so the image is the same. Floating-point sums would not be.
- **Caching only helps when increments repeat.** Accumulating locally and flushing once (EP3's per-thread `crossings`) needs many increments to the *same* address. The star image has ~377 M addresses in HD, and a thread keeps a few dozen photons, so a buffer has nothing to combine: it delays atomics but cannot remove them.
- **Techniques for when there is reuse.** Accumulate in a register (one counter); warp aggregation (`__match_any_sync`, one lane does the atomic); shared-memory privatization (a per-block histogram copy, how `cub::DeviceHistogram` works, needs the histogram to fit); sort then reduce (CUB radix sort plus run-length counts).
- **Possible experiment.** Time vs image size for global atomics vs a shared-memory copy, to show where privatization starts to win. **[to explore]**
- **Warp-level queue optimization.** One atomic per warp grabbing 32 chunks, handed out with `__shfl_sync`. **[to explore]**

## 5. Random numbers

- **What a PRNG is.** A deterministic function state → (new state, output). Quality (TestU01 BigCrush, PractRand) and, on a GPU, parallel streams.
- **C.** `rand()` is implementation-defined, has hidden global state and is not thread-safe. **`RAND_MAX` is 32767 on MSVC.** `rand()/RAND_MAX` can return exactly 0 and 1 (EP2). `drand48` is a 48-bit LCG; `erand48` takes your own state.
- **C++ `<random>`.** Engines (`mt19937`, ...) give the same sequence on every compiler, but distributions are implementation-defined, and none of it runs in device code.
- **cuRAND, device API.** `curand_init(seed, subsequence, offset, &state)` then `curand_uniform*`. Generators: XORWOW (the default, slow init because of the skip-ahead), Philox4_32_10 (counter-based, cheap init, used by EP3), MRG32k3a, MTGP32, Sobol (quasi-random).
- **cuRAND, host API and Thrust.** The host API fills GPU arrays (`curandGenerateUniform`), which costs memory bandwidth. `thrust::random` works on both sides but has weak generators.
- **Generators you can write yourself.** LCG, xorshift, xoshiro256\*\*, PCG32, SplitMix64, Philox4x32-10.
- **Giving every thread its own stream.** Different seeds (risky), skip-ahead / splitting, leapfrog, **counter-based**: `output = f(key, counter)`, no state to carry, any number in O(1). The main idea for the class. Reference: Salmon et al., *Parallel Random Numbers: As Easy as 1, 2, 3*, SC'11.
- **Bits to double, and the endpoints.** `(x >> 11) * 2^-53` gives [0,1). `curand_uniform` returns (0,1], `drand48` [0,1). Matters for `log(ξ)` and `cbrt(ξ)`.
- **Found on this machine.** **[measured]**
  - cuRAND's device API cannot be called from host code: nvcc says *calling a `__device__` function from a `__host__` function is not allowed*.
  - `curand_uniform_double` with Philox uses **one 32-bit draw**: `x * 2^-32 + 2^-32`, so only 2^32 distinct values in (0,1]. `curand_uniform2_double` combines two draws for 53 bits. (`curand_uniform.h`, CUDA 13.3.)
- **EP4 choice.** cuRAND Philox on the GPU, plus a host copy of cuRAND's Philox (same constants, counter layout and conversion) for the CPU path, checked bit for bit by a test kernel. **[decided]**
- **32 random bits are enough for the star.** Pixels, observer bins and spots are millions of times coarser than the 2.3·10⁻¹⁰ grid. EP4 uses cuRAND's standard 32-bit uniforms (`curand_uniform_double`, `curand_uniform`). Details in [precision.md](precision.md). **[decided]**
- **Where to read.** The CUDA C++ Programming Guide has almost nothing on RNGs. Read the cuRAND documentation (Device API overview, generator types, examples), the CUDA Samples (`MC_EstimatePiP`, `MC_SingleAsianOptionP`), and Random123.

## 6. Splitting the work: chunks

- **Chunk = unit of work and of randomness.** Photon i belongs to chunk `i / C`. Chunk c uses Philox stream c (`curand_init(seed, c, 0)`) and simulates its photons in order.
- **The chunk size must be fixed.** It belongs to the definition of the experiment. EP3's chunk size depended on the launch, so a different GPU or the CPU would draw different needles. EP4: a fixed default (1024), set with a flag. **[decided]**
- **Chunk range `[first, last)`.** One run simulates only those chunks. Partial images add up to exactly the full image, for any split.
- **What it makes possible.** The same photons at every SM count; horizontal scaling (containers); resuming long runs; cheaper N sweeps (the image for 10N = the image for N + more chunks).
- **Rules.** The same seed in every worker (split by range, never `seed + k`); spots come from their own reserved stream so every worker sees the same star.
- **Chunk size trade-off.** Too small: per-chunk overhead. Too large: small N does not fill the GPU (~55 000 threads), plus a tail at the end.

## 7. Performance, scaling and Amdahl

- **Amdahl's law with n = SMs.** T(1) is the GPU on one SM. CPU vs GPU is a separate throughput comparison, not part of Amdahl.
- **The serial part.** In EP3, a ≈ 0.15 s, mostly CUDA context creation (the first CUDA call), plus allocations and the SM-listing kernel. In EP4 it also includes zeroing the image, copying it back and writing it to disk.
- **p depends on N.** EP3: p ≈ 0 at N = 10⁴, 0.9986 at 3·10¹⁰. Amdahl's p is a property of the code *and* the problem size, which leads to Gustafson's law (weak scaling).
- **Gustafson's law.** Details: [scaling_laws.md](scaling_laws.md).
  - Scaled speedup S(n) = n − s′(n − 1), with s′ the serial share measured on the parallel run.
  - The problem grows with the machine, and the speedup grows linearly instead of hitting Amdahl's ceiling.
  - For EP3 at N = 3.2·10¹⁰ it gives 34.8 on 36 SMs (Amdahl, measured: 34.3).
  - For the star, the serial part grows with the image but not with N: Gustafson's assumption holds for one and not the other.
  - Planned experiments: strong and weak scaling over SMs, weak scaling over the image ladder (512 → 1k → 2k), and the largest problem each machine fills in a fixed time. **[to measure]**
- **Image resolution also changes p.** A bigger image means more serial memset, copy and write time.
- **Timing with three clocks.** Wall time, kernel time, and `clock64()` cycles (independent of laptop boost: 2017–2908 MHz in EP3). **[measured]**
- **Precision.** Consumer GPUs run FP64 at 1/64 of the FP32 rate: EP3 FP32 was 16× faster than FP64. FP16 was no faster (conversions, `hsin`, cuRAND makes floats). FP32 stops gaining accuracy near 7 digits. **[measured]** Whether the star needs FP64 is in section 12.
- **Vertical vs horizontal scaling.** Vertical = more resources for one program (SMs, CPU threads), coordinated through shared memory and atomics. Horizontal = more independent copies (containers, MPI), coordinated by splitting the work up front or by messages. The comparison is about the techniques. **[to explore, optional]**
- **Container notes.** `--cpus` sets a CFS quota (throttled time slices, noisy). `--cpuset-cpus` pins to cores (what Amdahl needs). The Core Ultra 7 255HX mixes P-cores and E-cores. Containers on one laptop share memory bandwidth, cache, power and heat. Docker runs inside a WSL VM with its own limits (`.wslconfig`).

## 8. The star problem (EP2 → EP4)

- **Observers are bins on a sphere.** nmu equal steps in μ = cos θ and nphi equal steps in φ. Equal μ steps mean equal solid angles, so every bin gets 1/(nmu·nphi) of the photons. The nphi bins of one latitude are the frames of the rotation video.
- **Planning formula.** Counts per pixel ≈ N / (nmu · nphi · (π/4) · nx · ny). SNR = √counts, so 100 counts give SNR 10.
- **Resolution table.**
  - EP2: 200², 150 φ, 150 μ → 96 MB, 7·10¹⁰ photons.
  - HD: 512², 360 φ, 180 μ → 1.5 GB, 1.3·10¹² photons.
  - Max: 1024² → 6 GB, 5.3·10¹² photons.
  - Image memory = nx·ny·nphi·n_observers·4 bytes, against 8 GB of VRAM.
- **Binning waste.** Only n_observers/nmu (~2%) of photons are recorded. That makes the atomics rare and cheap on the GPU.
- **Observer table.** `observer_of_imu[imu]` (−1 means not an observer) replaces EP2's search loop. It lives in constant memory.
- **If the image does not fit in VRAM.** Split by observers: run the same chunks again and record a different subset of observers each time.
- **uint32 per pixel is enough.** A pixel would need 4·10⁹ counts to overflow.
- **Peel-off / forced observation.** Every photon contributes to every observer with a weight. No waste, no angular blur, but ~1440 atomics per photon and float weights. Not the professor's algorithm: kept for later, maybe. **[to explore]**
- **EP2 issues to fix in the port.** `Image.N` is `unsigned int` (overflows above 4·10⁹), `cos()` is recomputed for every photon and spot, `rand()`.

## 9. Tools

- **Nsight Systems (`nsys`).** A timeline of the whole program: CUDA API calls, copies, kernels, idle gaps. Shows the serial part as a picture. `nsys profile -o name --stats=true ./prog`.
- **Nsight Compute (`ncu`).** One kernel in depth.
  - Speed Of Light: % of peak compute vs memory, which says whether the atomics matter.
  - Roofline, occupancy, warp stall reasons, memory workload (L2 atomics).
  - Source view per line (compile with `-lineinfo`).
  - It replays kernels many times, so profile a small N. `ncu --set full -k kernel -c 1 ./prog`.
- **`compute-sanitizer`.** memcheck (out-of-bounds, e.g. the image index), racecheck, initcheck.
- **`cuda-gdb`.** Step through device code (compile with `-G`).
- **Installed here.** **[measured]** Nsight Systems 2026.1, Nsight Compute 2026.2, compute-sanitizer and cuda-gdb (CUDA 13.3) in the `aga511` WSL distro.
- **WSL notes.**
  - GPU performance counters need permission on Windows (NVIDIA Control Panel / App → Developer → Manage GPU Performance Counters → allow all users), otherwise `ERR_NVGPUCTRPERM`.
  - Some nsys system-level features are limited under WSL.
  - The Windows GUIs open the report files.
- **First profile: EP3's Buffon in `nsys`.** **[measured]** Full notes for the class: [`../cuda_class/nsight.md`](../cuda_class/nsight.md).
  - The kernel took 1.4794 s on the GPU's clock vs 1.4795 s on EP3's own timer: the cheap timer is right.
  - The serial part (~0.15 s) is CUDA context creation: 154 ms between the first CUDA call and the first `cudaMalloc`, ~400 `ioctl` calls into the driver. Allocations, copies and launches together take ~3 ms.
  - `nsys` added ~0.4 s at start-up but did not change the kernel time.
  - `ncu` failed with `ERR_NVGPUCTRPERM`: the counters must be enabled on Windows first. **[to do]**
- **"Don't record" switch.** Run the same photons without writing the image, to measure what recording costs. **[decided]**

## 10. Libraries

- **cuRAND.** Random numbers (section 5).
- **CUB.** Block, warp and device primitives: reductions, scans, radix sort, `BlockHistogram` / `DeviceHistogram`. The candidate for smarter image accumulation. **[to explore]**
- **Thrust.** STL-like algorithms on GPU arrays (`reduce`, `transform`), handy for totals and checks.
- **CUDA events.** `cudaEventRecord` / `cudaEventElapsedTime` for timing on the GPU's own clock.

## 11. Building

- **Separate files and device code.** A `__device__` function in one `.cu` cannot be called from a kernel in another `.cu` unless you compile with `-rdc=true` (relocatable device code: slower linking, less inlining). The usual fix: physics and RNG in headers (`.cuh`) as `static inline __host__ __device__` functions. **[decided: EP2-style split files]**
- **Compile flags.** `nvcc -O2 -arch=native`; `-lineinfo` for profiling; `-G` for debugging.

## 12. Floating-point precision

Full notes: [precision.md](precision.md).

- **Formats.** FP64 has 53 bits of precision (~16 digits), FP32 24 bits (~7), FP16 11 bits (~3, largest value 65 504).
- **Random bits vs type.** `curand_uniform_double` returns a double built from one 32-bit draw. `curand_uniform2_double` gives 53 bits. **[measured: read from the CUDA 13.3 headers]**
- **Monte Carlo is forgiving.** Each photon is a short independent calculation and the tally is an integer, so rounding errors do not accumulate. Only a bias inside one photon could matter.
- **Where precision bites.**
  - Cancellation in `sqrt(1 - z*z)` near the poles: use `(1 - z)(1 + z)`.
  - Comparing cosines near 1 for small spots: use the squared chord.
  - The precision needed depends on the smallest physical scale.
- **FP16 cannot represent the image geometry.** Its spacing is ~10⁻³ vs a 2·10⁻³ pixel.
- **EP4.** The physics is written on a `real` type (FP64 by default, FP32 with `-DSTAR_FP32`), with FP64 as the reference. FP32-friendly formulas are used from the start. FP32 vs FP64 is to be measured (χ² per pixel and speed). **[decided]**
- **Prediction.** The FP64/FP32 speed gap will be bigger for the star than for Buffon (closer to 64×), because the star uses more FP64 math functions. **[to measure]**
- **`sine_from_cosine`.** **[measured]** Details in [precision.md §8](precision.md#8-deep-dive-sine_from_cosine-and-the-fp32-correction).
  - `(1 − c)(1 + c)` vs `1 − c²` in FP32: worst case 1.0·10⁻⁷ vs 4.3·10⁻⁵.
  - In the star the two give the same images.
  - The real FP32 hot spot is the projection at the pole observers (~30× more photons change pixel), a coordinate singularity.
- **On the IAG A30, FP64 costs only ~2× FP32** (vs 64× on the laptop). The precision trade-off flips between GPUs. See [gpus.md](gpus.md). **[to measure]**
- **First measurement, CPU, N = 10⁸.** **[measured]**
  - Same photons in FP32 and FP64; 5·10⁻⁴ of them land in a different pixel or bin.
  - The Poisson noise per lit pixel is 0.22, hundreds of times larger.
  - FP32 is about 1.6× faster on one CPU core.

## 13. EP4, sequential engine: what is built and checked

Code and usage: [`../EP4/README.md`](../EP4/README.md).

- **Layout.** EP2-style split files built with `nvcc`, with no device code yet:
  - `real.cuh`, `rng.cuh`, `geometry.cuh`, `photon.cuh` (headers with `static inline HD` functions);
  - `scene`, `image`, `options`, `main` (`.cu` files).
- **Code style.** Self-documenting names, small functions, almost no comments; the reasoning lives in these docs.
- **Physics is pure.** `trace_photon` returns a pixel index or a reason the photon is lost, and writes nothing. It is templated on the random generator, so on the GPU it takes cuRAND's state.
- **Checks.** **[measured]**
  - Philox matches the Random123 test vectors.
  - Two halves of the chunk range add up to the full image bit for bit.
  - EP4 vs EP2 without spots: χ²/dof = 1.0047 over 126 008 pixels.
  - EP4 runs at 1.0·10⁷ photons/s in FP64 on one core (EP2: 7.6·10⁶ including setup and save).
- **`calloc` vs `malloc` + `memset`.** `calloc` of a big block gets lazily zeroed pages from the OS: the zeroing happens on first touch, inside the photon loop, and would be counted as parallel time. `memset` touches every page during setup, so it is counted as serial time, where it belongs. A detail that matters for Amdahl's law.
- **Checking the observer before the pixel.** ~97% of photons fly toward no observer, so testing that first skips the pixel computation for most of them.
- **Not done yet.** Bit-for-bit check of the CPU Philox against cuRAND (needs the GPU step); the `--no-record` experiment.

## 14. Ideas for the class

- Build the talk the way the code is built: host/device → hierarchy → memory → atomics → occupancy, with Buffon as the running example and the real Amdahl measurement from this GPU at the end.
- Live demo: `nsys` and `ncu` on EP3's `buffon`.
- Slides worth making:
  - counter-based RNG (one stream per chunk);
  - the cuRAND 32-bit double finding;
  - "p depends on N";
  - the FP64/FP32 gap;
  - the CPU copy of cuRAND's Philox checked bit for bit.

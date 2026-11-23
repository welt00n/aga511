# The CUDA Programming Guide, as a reference for the class

NVIDIA's [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html), v13.5 (updated 2026-10-31). Read on 2026-11-23 for the class: what it says on the points the class makes, with the section to cite, and how each compares with our deck and our code. Quotes are short; the guide is the source.

## How the guide is organized (CUDA 13 edition, restructured)

| Part | What | Sections the class uses |
|---|---|---|
| 1. Introduction | the model, language-agnostic | 1.1 Introduction (history, CPU vs GPU), 1.2 Programming Model (threads, blocks, grids, warps, SIMT, memory), 1.3 The CUDA platform (compute capability, PTX, cubin) |
| 2. Programming GPUs in CUDA | the basics, C++ and Python | 2.1 Intro to CUDA C++ (kernels, launch, memory, sync, errors), 2.3 Writing SIMT Kernels (memory spaces, atomics, occupancy), 2.7 NVCC |
| 3. Advanced CUDA | finer control | 3.2 Advanced Kernel Programming (hardware multithreading, warps, independent thread scheduling) |
| 4. CUDA Features | complete feature references | 4.4 Cooperative Groups (cooperative launch, grid sync) |
| 5. Appendices | reference | 5.1 Compute Capabilities (limits per SM), 5.6 Floating-Point Computation (IEEE 754, FMA, ULP tables) |

New in this edition: tile programming (a second model next to SIMT, where the compiler picks the threads per block), locality domains, and the statement that texture memory gives no advantage outside graphics.

## What it says, point by point

### History and why GPUs (1.1)
- 1.1.1: born for 3D graphics; "By 2003, some stages of the graphics pipeline became fully programmable"; "In 2006, NVIDIA introduced" CUDA. (G80 is not named.) **Deck:** "2006–2007: CUDA" is consistent (announced with the G80 in 2006, the toolkit released in 2007).
- 1.1.2: a CPU excels at "a serial sequence of operations"; a GPU at "thousands of threads in parallel", trading single-thread speed for throughput; GPUs "devote more transistors to data processing units", CPUs to "data caching and flow control". **Deck:** the CPU vs GPU slide says the same.

### Threads, blocks, grids, scheduling (1.2.2, 2.3.7)
- Warps are "groups of 32 threads"; a block holds up to 1024 threads (2.1.2.2.1, 5.1).
- 1.2.2.1: "All threads of a thread block are executed in a single SM"; blocks "may be executed in any order"; "There is no guarantee of scheduling between thread blocks".
- 2.3.7: the scheduler assigns blocks to SMs while they have resources, otherwise it waits for blocks to finish; "no ordering guarantees are made by the scheduler".
- 3.2.2.2: if an SM can't hold one block, "the kernel will fail to launch".
- **Deck:** "Os blocos vão inteiros para um SM, e em qualquer ordem" says exactly this; cite 1.2.2.1.

### SIMT and divergence (1.2.2.2, 3.2.2.1)
- "A warp executes one common instruction at a time"; on a divergent branch "the warp executes each branch path taken, disabling threads that are not on that path" (3.2.2.1). SIMT "does not have a fixed data-width like SIMD" (1.2.2.2).
- Independent thread scheduling since compute capability 7.0 (one program counter per thread; use `__syncwarp()` for warp-synchronous code).
- 3.2: SMs "issue instructions in order and do not perform branch prediction or speculative execution".
- **Deck:** the divergence slide matches; independent thread scheduling is a remark for questions.

### Memory spaces (2.3.3, Table 1)

| Space | Scope | Lifetime | Where |
|---|---|---|---|
| register | thread | kernel | SM |
| local | thread | kernel | device memory |
| shared | block | kernel | SM |
| global | grid | application | device memory |
| constant | grid | application | device memory (cached) |

- Local memory: "the physical location of local memory is in the global memory space"; same latency and bandwidth as global; used for non-constant-indexed arrays, large structures and spills (2.3.3.4). **Deck:** "a memória 'local', que mora na memória do device, atrás da L1" is right.
- Constant: "typically 64KB per device"; constant cache working set 8 KB per SM (5.1). **Deck and code:** the star (18 048 B) is in constant memory.
- Texture "do not provide any performance advantage for non-graphics applications" (2.3.3.7).

### The basic program (2.1)
- Kernels: `__global__`, `void`; launch `<<<grid, block>>>`; index `threadIdx.x + blockDim.x * blockIdx.x`, "very common for 1-dimensional parallelizations", with a bounds check; blocks `(n + threads - 1) / threads` or `cuda::ceil_div`.
- "Kernel launches are asynchronous with respect to the host thread" (2.1.4); "The `cudaMemcpy` API is synchronous" (2.1.3.2); `cudaDeviceSynchronize` waits for all work.
- Error checking: "best practice to always check" the return of every API call (2.1.7). **Code:** the engine's `check(...)` around every call.
- The grid-stride pattern appears as code (2.3.3.8: `i += blockDim.x * gridDim.x`); the guide doesn't name it. **Deck:** our slide uses the same step.

### Synchronization and atomics (2.3.3.2, 2.3.5, 2.3.6, 5.4.5)
- `__syncthreads()` "blocks all threads in the thread block until all threads have reached the call".
- The model "does not provide mechanisms or guarantees that enable synchronization across thread blocks" (2.3.6); blocks share results through atomics (2.1.5).
- Histograms: compute in shared memory per block, then "global memory atomics" (2.3.3.8). **Code:** the EP4 image is a histogram of photons, with global atomics directly (5 GB doesn't fit in shared memory).
- "The ordering of arithmetic atomic operations depends on hardware scheduling" (5.6.8): float sums done by atomics are not reproducible. **Code:** the image and the tallies are integers, so their atomic sums are exact in any order. That is part of why the image is bit for bit.

### Occupancy (2.3.7)
- Occupancy is "the ratio of the number of active warps to the maximum" per SM.
- "it's a good practice to have occupancy as high as possible which hides latency and increases performance"; limiting registers "may result in a net increase in performance".
- **Our measurement disagrees here, and that is a teaching point:** on the laptop, capping registers to raise occupancy is never faster (FP64 flat to 0.67, then up to 10% slower; FP32 fastest uncapped), because the kernel is bound by its arithmetic units, not by latency. The guide's advice is a default, and the measurement decides.
- The guide's occupancy example uses a 768-thread block, the same as our FP64 block.

### Cooperative launch (4.4.8)
- `cudaLaunchCooperativeKernel` "ensures that the kernel launch is atomic": if the call succeeds, all the blocks launch together; supported from compute capability 6.0, queried with `cudaDevAttrCooperativeLaunch`. Needed for grid-wide sync.
- The guide gives no sizing formula; the occupancy API (`cudaOccupancyMaxActiveBlocksPerMultiprocessor` × SMs) is what our engine uses. Too large a grid fails with `cudaErrorCooperativeLaunchTooLarge` (runtime API reference, not the guide).
- The guide doesn't use the words "persistent threads". **Deck:** cite CS149 for the name and the guide for the guarantee.

### The limits of our two GPUs (5.1)

| | 8.0 (A30) | 12.0 (laptop) |
|---|---|---|
| warps / threads / blocks per SM | 64 / 2048 / 32 | 48 / 1536 / 24 |
| registers per SM; per thread | 64 K; 255 | 64 K; 255 |
| shared memory per SM; per block (opt-in) | 164 KB; 163 KB | 100 KB; 99 KB |
| L1 + shared | 192 KB | 128 KB |
| constant memory; constant cache per SM | 64 KB; 8 KB | 64 KB; 8 KB |
| FP32 : FP64 throughput | 2 : 1 | 64 : 1 |

The per-SM rows match `research/sm_limits.py` on the laptop and `KNOWN_TARGETS["80"]` in `lab/device.py`. The guide doesn't give the register allocation granularity (we use 8 per thread, from `sm_limits.py`), nor the FP32/INT32/FP64 units per SM (those come from the whitepapers).

### Floating point (5.6)
- "All GPU devices follow the IEEE 754-2019 standard", without dynamic rounding modes or exception flags.
- Contraction into FMA is on by default (like `--fmad=true`); `--fmad=false` turns it off. **Code:** the `no_fma` variant.
- Host and device math libraries "are independent"; differences "do not necessarily mean the GPU's result is incorrect". ULP bounds: `sinf`/`cosf` 2, `atan2f` 3, `cbrtf` 1; double `sin`/`cos`/`atan2` 2, `cbrt` 1; only basic operators and `sqrt` are exact.
- **Deck:** this is the explanation for "FP64 the same bits, FP32 not quite": in FP64 a 2-ULP difference almost never moves a photon across a pixel edge (none in 10⁸); in FP32 some do (2.98·10⁻⁴ of the photons).

### The platform (1.3)
- Compute capability = the SM version (12.0 → `sm_120`); PTX is a "virtual instruction set architecture" (`compute_80`); a cubin is for one SM version; PTX can be JIT-compiled for any later compute capability.
- **Research link:** our A30 build takes 96 registers because the PTX for `compute_80` comes from the older front end (`research/front_end_registers.py`); the guide's PTX/cubin split is the vocabulary for that story.

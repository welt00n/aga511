# Concepts used in EP4 (notes for the class)

Running notes, filled while EP4 is built: each concept we use, where EP4 uses it, and what we measured. Sunday's class prep starts here. The class arc (blocks 0–10) is in [class_plan.md](class_plan.md); the deeper explanations are in [../docs/](../docs/README.md).

**Columns:** *Concept*; *In EP4* (file or step); *Measured / shown* (a number or a result the class can see); *Block* (where it fits in the class arc).

## Already used (phases B and C, before the GPU)

| Concept | In EP4 | Measured / shown | Block |
|---|---|---|---|
| Counter-based RNG (Philox4x32-10): a stream per chunk, no state shared between threads | `rng.cuh`, one subsequence per chunk | Random123 known answers 3/3; split runs = full run, bit for bit | 6, 8 |
| Our own CPU copy of cuRAND's Philox (cuRAND's device API cannot run on the host) | `rng.cuh` | same numbers as cuRAND (checked on the GPU in step 2) | 8 |
| Chunks: the unit of work, so any split gives the same image | `main.cu`, `--first-chunk/--last-chunk` | two halves = the full run, bit for bit | 5, 6 |
| A pure photon: the physics returns a pixel and writes nothing | `photon.cuh` (`trace_photon`) | the same code will run on CPU and GPU (`HD`) | 3, 8 |
| `__host__ __device__` (`HD`), templates for the RNG type | `real.cuh`, `photon.cuh` | one physics, two RNGs | 3 |
| FP64 vs FP32 (53 vs 24 bits), `sine_from_cosine`, chord tests | `real.cuh`, `photon.cuh` | CPU: FP32 1.5× faster, 1.2·10⁻⁴ of photons move pixel | 7, 9 |
| Equal solid angles: bins in μ = cos i (Archimedes) | `observers.cu` | every observer gets the same share of photons | 8 |
| The Scene: computed once, read-only, small (for constant memory) | `scene.cu` | spots, thresholds, μ-bin → slot table (16 KB) | 4, 8 |
| Memory budget → passes: memory traded for time | `options.cu`, `main.cu` | 4 passes ≈ 3.7× one pass; passes = one pass, bit for bit | 8 |
| Image writes cost more than the arithmetic around them | `count_photon` | ~180 ns per recorded photon on the CPU; pixel math not measurable | 8 |
| First touch: `malloc` reserves, the first write maps the page | `image.cu` (`touch_every_page`) | pass 1 zeroing 25–30× slower until pages were touched in setup | 4 |
| The compiler can remove work you meant to time (`malloc` + `memset(0)` → `calloc`) | `image.cu` | SETUP 0.0005 s instead of 0.15 s | 7 |
| Phase timings = the serial and parallel parts of Amdahl's law | `report.cu` (`SETUP`, `ZERO`, `PHOTONS`, `SAVE`) | measured, not only fitted | 7 |
| Safe output: `.partial`, `fsync`, rename | `files.cuh` | `fsync` 0.31–0.43 s per 315 MB | (aside) |
| Amdahl vs Gustafson | [scaling_laws.md](../docs/scaling_laws.md) | EP3: 34.3 measured vs 34.8 Gustafson | 7 |
| Buffon's serial part is CUDA context creation | [nsight.md](nsight.md) | ~0.15 s with `nsys` | 7 |

## The frame: the hardware is the architecture ([hpc_architecture.md](../docs/hpc_architecture.md))

| Concept | In EP4 | Measured / shown | Block |
|---|---|---|---|
| The hierarchy: node → processor → compute units → lanes, plus memory spaces | the file names (`cpu.cu`, `cpu_cores.cu`, `gpu.cu`, `gpu_sms.cu`) | each level is a scaling axis | 2, 3 |
| Execution space vs memory space (Kokkos' terms) | `processor.cuh`; `Image` vs `GpuImage` | — | 4 |
| Hybrid CPUs: 8 P-cores + 12 E-cores, no hyper-threading | the laptop's CPU | WSL shows 20 identical CPUs: the hardware is hidden from us | 2, 7 |
| GPU clocks change: 825 MHz idle, 3090 MHz boost | the laptop's GPU | report clocks with every timing | 7 |
| Fair benchmarking: state the baseline of every speedup (Hoefler & Belli 2015, after Bailey 1991) | `measure`'s `speed` part: the GPU against itself, one SM vs all SMs | the whole GPU vs one CPU core was dropped as unfair (2026-11-22, below) | 7, 10 |
| The roofline: compute-bound vs memory-bound | FP64 (1:64) vs FP32 (atomics) | to draw at the board | 2, 9 |
| Where each line runs: integer units (Philox), FP64 units vs FP32 cores, the special function unit (`MUFU`), L2 atomics (`RED`), the constant cache | [hpc_architecture.md §7](../docs/hpc_architecture.md#7-the-code-mapped-to-where-it-runs-the-gpu-one-photon) | EP3's kernel: IMAD 302, LOP3 209, IADD 136 (Philox is integer work), DFMA 13, FFMA 19, HFMA2 23 | 3, 8 |
| Reading the compiled code: SASS, `cuobjdump`, the compiler's resource report | `instruction_mix.py`, `nvcc -Xptxas -v` | EP3's kernel: 58 registers, 72-byte stack, no spills | 3 |
| Constant memory: broadcast (one address per warp) vs serialized (a different address per thread) | spot centers (broadcast) vs `slot_of_imu` (serialized?) | experiment: constant vs global memory | 4, 8 |
| Warp-aggregated atomics, done by the compiler | `SHFL`, `VOTEU`, `UPOPC` next to the queue's `atomicAdd` in EP3's SASS | to confirm in EP4's | 5 |
| Static vs dynamic instruction counts | `instruction_mix.py` vs the section timers | the static mix shows *which* units, the timers *how much* | 7 |
| Measuring without a profiler: CUDA events, `clock64()` section timers, per-SM counters | `star_gpu64_sections` | cycles per piece of the photon, FP64 vs FP32 | 7 |
| The observer effect: measuring changes what is measured | the section build vs the normal build's kernel time | the difference is shown | 7 |

## Today (the GPU), filled as we go

| Concept | In EP4 | Measured / shown | Block |
|---|---|---|---|
| One pass loop, two processors: an interface (`processor.cuh`), chosen at link time | `cpu.cu`, `gpu.cu`; `star_cpu64` vs `star_gpu64` | the CPU checks unchanged after the split | 3 |
| An opaque type: hold a pointer, never look inside | `typedef struct Processor Processor;` | — | 3 |
| What `nvcc` does: one call, preprocess + compile + link; the build is just a command | `engine.py` prints each `nvcc` command | 4 s per CPU binary; 0 s when the code hash matches | 3 |
| SASS only: the binary carries machine code for one compute capability | `cuobjdump --list-elf --list-ptx build/star_gpu64` | `sm_120` cubins, no PTX | 1, 3 |
| Asking the GPU about itself | `cudaGetDeviceProperties` in `gpu_device.cu` | RTX 5070 Laptop, cc 12.0, 36 SMs | 2 |
| cuRAND's device API: `curand_init(seed, subsequence, offset)`, `curand`, `curand_uniform(_double)`; header-only, device-only | `gpu_rng.cuh` | — | 6, 8 |
| The same random numbers on CPU and GPU, bit for bit (counter-based RNG) | `--test-rng` on `star_gpu64` | identical for chunks 0, 1, 10⁶ + 3, 2³² + 5, FP64 and FP32 | 6, 8 |
| Overloading + templates: one physics, two generators | `uniform(PhiloxState *)` vs `uniform(GpuRng *)` | — | 3 |
| A first kernel launch: `<<<1, 4>>>`, `cudaMalloc`, `cudaMemcpy`, `cudaGetLastError` | `draw_first_numbers` | 19 registers | 3, 4 |
| Name collisions from copying a library (macros are global) | `PHILOX_W32_0` vs cuRAND's | a compiler warning, renamed | (aside) |
| The kernel: persistent threads + the work queue, the same as Buffon's; only the chunk's body changes | `trace_photons` vs EP3's `throw_needles` | — | 6, 8 |
| Constant memory for the read-only Scene (`__constant__`, `cudaMemcpyToSymbol`) | `gpu_scene`, ~18 KB, loaded per pass | — | 4, 8 |
| Atomics on the image in device memory | `count_photon_atomic` (`atomicAdd`) | — | 5, 8 |
| Tallies in registers, one atomic per counter per thread at the end | `add_tallies` | — | 5 |
| Pinned host memory: the copy engine writes straight into it | `cudaMallocHost`, one observer's worth | — | 4 |
| Registers per thread decide occupancy; FP64 needs more | ptxas report | FP64 78, FP32 67 (Buffon 58), no spills | 3, 8 |
| **Bit for bit vs statistically equal** | `--compare-gpu` | **FP64: the GPU's image = the CPU's, bit for bit**; FP32: 3.2·10⁻⁵ of the photons move | 8, 9 |
| Fused multiply-add changes the last bit | `--fmad=false` experiment | FP32 differences 54 → 12 pixels: FMA is 3/4 of them, the math library the rest | 7, 9 |
| The first speedups, with the baseline stated: **the whole GPU vs one CPU core**, photon phase vs whole run | `--compare-gpu` (prints both) | FP64, N = 10⁸: 75× on the photons, ~22× on the whole run; FP32 is 14.6× FP64 on this GPU (1:64) vs 1.45× on the CPU | 7, 9 |
| Amdahl in one run: a fast kernel next to a fixed setup | N = 10⁸ FP32: photons 0.011 s, setup 0.14 s | 90% of the run is serial | 7 |
| `%smid`: a thread can ask which SM it runs on; the SM mask | `gpu_sms.cu` (EP3's) | the same image on 1, 2, 9, 36 SMs | 6, 7 |
| Cooperative launch: every block resident at once | `run_kernel` | — | 6 |
| Load balance from a work queue: fast SMs take more chunks | `SM` lines | chunks per SM within 0.1%, cycles within 0.2% | 5, 7 |
| **The GPU throttles its clock under load** (power, temperature) | `CYCLES` / time | **2.88 GHz on 1 SM, 1.95 GHz on 36**: speedup 30.7× in cycles, 20.8× in seconds | 7 |
| Measure in cycles and in seconds, and report both | `clock64()` per SM next to the host's clock | the clock ratio explains the gap exactly | 7 |
| One SM vs one CPU core | `--sms 1` vs `star_cpu64` | FP64 1.25×, FP32 ~25× | 2, 7 |
| **Occupancy limited by registers:** 65 536 registers per SM ÷ registers per thread | `REGISTERS_PER_THREAD`, `OCCUPANCY` | FP64: 80 → 768 threads = 0.50; FP32: 67 → 896 = 0.58; one block per SM | 3, 8 |
| The kernel's resources from the runtime: `cudaFuncGetAttributes` | `KernelResources` | 40 B stack, 18 KB constant (the Scene) | 3, 4 |
| Asking the GPU: free memory, L2, bandwidth, FP32:FP64, versions | `cudaMemGetInfo`, device attributes | 7.34 GB free, L2 32 MB, ~380 GB/s, 1:64, CUDA 13.3 / driver 13.5 | 2, 4 |
| The memory budget from the hardware: passes chosen to fit | `fit_observers_per_pass` | 10 GB job → 4 + 4 observers in 2 passes | 4, 8 |
| Device → host copy into pinned memory | `COPY_TIME` | ~24 GB/s | 4 |
| The serial part grows with the image: the disk | 1k, 8 observers | save 20 s vs photons 0.12 s | 7 |
| **What the kernel is made of:** integer work is the largest instruction class (Philox, indices) | `instruction_mix.py` | FP64 kernel: 660 integer, 278 FP64, 73 FP32, 13 MUFU, 2 atomic sites | 3, 8 |
| **Compute-bound vs memory-bound, measured:** constant vs global memory makes no difference | `scene_in_global` variant | 1.002× (FP64), 0.984× (FP32) | 4, 8 |
| **Occupancy is not a goal in itself:** capping registers fills the SM but spills | `registers_64`, `registers_40` | occupancy 1.0 → 9% slower (FP64) | 3, 8 |
| Where the cycles go, per piece of the photon | `sections` variant | direction 33–41%, surface 20–28%, atan2 22% (FP64), atomics 13–15% | 8, 9 |
| Amdahl inside the kernel: only the FP64 part slows down | FP64 vs FP32 per photon | 17×, not 64× | 7, 9 |
| The clock depends on the work (power limit) | `gpu_monitor.py` | FP32 2310 MHz at 72 W, FP64 2812 MHz at 55 W | 7 |
| **Ask the driver why the clock is low:** `clocks_event_reasons` | `nvidia-smi --query-gpu=…` under load | FP64, or 1 SM: the voltage ceiling (`reliability`) at 2880–2895 MHz; FP32 on 36 SMs: `sw_power_cap`, with the enforced limit moving 85–115 W in one run (the laptop shares power between CPU and GPU). Not the spills | 7 |
| **p(N): Amdahl's parallel fraction depends on the problem size** (EP3's result, for the star) | `problem-size` sweep, `scaling.py` | FP32 photon phase: 0.14 (10⁵) → 0.9993 (10⁹); whole run: ≈ 0 → 0.987 (10¹⁰) | 7 |
| Speedups at the same precision, the baseline named | `devices` sweep (2026-11-21) | GPU vs one core: 82× (FP64), 850× (FP32) on the photons; ~28× on the whole run at 10⁸. **Withdrawn:** see "the unfair comparison" below | 7, 9 |
| **The unfair comparison** (Lee et al., *Debunking the 100X GPU vs. CPU myth*, ISCA 2010): a whole GPU against one core of an untuned sequential program credits the GPU with every core the CPU didn't use | our own "70× / 777× one core" (2026-11-22) | like for like it was 2.0× (FP64) and 23.6× (FP32) per SM vs per core, and ≥ 3.5× / ≥ 39× against a 20-core bound. Now the unit is the SM: all 36 SMs = **35.2× one SM** (FP64), **32.9×** (FP32; 35.9× in cycles). The CPU checks correctness only | 7, 9, 10 |
| **Strong scaling over SMs, measured:** FP64 35.9× on 36 SMs, p = 0.9998 | `strong_fp64` | the clock steady at 2880 MHz | 7 |
| **The clock in the speedup:** FP32 36.0× in cycles, 33.1× in seconds | `strong_fp32` | 2895 → 2636 MHz on 36 SMs | 7 |
| **Weak scaling and Gustafson, measured:** flat time, s′ fitted 0.054 vs measured 0.058 (FP64) | `weak_fp64` | — | 7 |
| Gustafson's scaled speedup: one processor's time for the *n-fold* problem / n processors' time | `scaling.py` | my first formula counted the serial part n times (negative s′): corrected | 7 |
| **Amdahl's ceiling, measured:** at N = 10⁸, 36 SMs give 2.3×; p = 0.58, ceiling 2.4 (the CUDA context is the wall) | `amdahl_ceiling_fp32` | the class's central plot: three curves, one GPU | 7 |
| **The showcase:** 3.9·10¹² photons in 6.4 min on the laptop's GPU (~4 days on one CPU core) | `showcase.py` | 1k, 300 frames, 4 observers, one pass | 9 |
| Planning from physics, checked by measurement: N ↔ SNR | `showcase.py` | SNR measured = predicted: 14.1 center, 11.7 limb ring | 9 |
| Reading a 1.3 GB file frame by frame (`np.memmap`) | the video writer | never loaded whole | (aside) |
| **The baseline itself varies:** the one-core CPU rate differs by ~25% between sessions | the `devices` sweep vs re-runs on 2026-11-22 | 7.0–8.0·10⁶ in the sweep, 8.8–9.1·10⁶ in every re-run (each vCPU, long runs, after the GPU, with `nvidia-smi`); cause not found; "shared power" not supported. A speedup is only as good as its baseline (82× vs ~65×) | 7, 9 |
| **`NOP`s are waits:** the compiler makes a warp wait for a slow FP64 result | `cuobjdump -sass` of the FP64 kernel | 1063 `NOP`s, in runs of 4 between dependent FP64 instructions; other warps run meanwhile: latency hiding, and part of why FP64 is 17× and not 64× | 3, 8 |
| Static vs dynamic instruction counts | `instruction_mix.py` | 4 FP64 instructions in the FP32 kernel are `sinf`/`cosf`'s slow argument reduction, never run for angles in [0, 2π] | 8 |
| Short kernels cannot be timed | Q3 re-runs | ~30 ms kernels vary ±30% run to run; ~170 ms ones within 1%: the "5× slower pass" was one noisy sample | 7 |
| FMA: faster and (in FP32) a different last bit | `no_fma` variant | FP64 +7% speed, identical images; FP32 30 689 pixels differ at 5·10⁹ | 7, 9 |

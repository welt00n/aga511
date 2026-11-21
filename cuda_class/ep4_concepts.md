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
| Fair benchmarking: state the baseline of every speedup (Hoefler & Belli 2015, after Bailey 1991) | `devices` sweep: vs one core and vs all cores | — | 7, 10 |
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

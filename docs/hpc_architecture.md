# EP4 as an HPC project: the hardware is the architecture

**Decided 2026-11-21:** the code's structure follows the hardware's (option B of [gpu_engine.md §9](gpu_engine.md#9-decisions)). **Status: brainstorm, to review.**

## 1. The idea

An HPC program is a map of a machine: a **node** has **processors**, a processor has **compute units**, a compute unit runs **lanes**, and data lives in **memory spaces** next to them. If EP4's files and names follow that map, three things follow:

- **The code reads like the hardware:** `gpu_sms.cu` is the GPU's SMs; `cpu_cores.cu` the CPU's cores.
- **Every level is a scaling axis we can measure:** nodes, processors, compute units, the launch.
- **The class's story is the hardware's story:** history → the chip → the hierarchy → our code, at each level.

This is how the established HPC frameworks think, too: **Kokkos** separates *where code runs* (an **execution space**: Serial, OpenMP, CUDA…) from *where data lives* (a **memory space**: host RAM, device memory…), and each execution space has a default memory space ([Kokkos programming model](https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/ProgrammingModel.html), [execution spaces](https://kokkos.org/kokkos-core-wiki/_sources/API/core/execution_spaces.rst.txt)). Our `processor` is an execution space; `Image` and `GpuImage` are memory spaces.

## 2. The map

| Level | Laptop | IAG node | In EP4 | The scaling question |
|---|---|---|---|---|
| **Node (machine)** | the laptop | the A30 server | the driver's `machines/` profiles; `remote` | the same job on two machines (Gustafson across machines); chunk ranges let two nodes split one job, like MPI without MPI |
| **Processor** | Intel Core Ultra 7 255HX; RTX 5070 Laptop GPU | its CPU (to see); NVIDIA A30 | `processor.cuh`: `cpu.cu`, `gpu.cu` | CPU vs GPU, stated against the best CPU run (section 4) |
| **Compute units** | **20 cores** (8 P-cores + 12 E-cores, no hyper-threading); **36 SMs** | ? cores; **56 SMs** | `cpu_cores.cu` (threads, one per core); `gpu_sms.cu` (the SM mask) | **Amdahl and Gustafson over compute units**, for both processors |
| **Lanes** | SIMD (unused: the photon loop branches); **warps of 32 threads**, 1536 threads per SM | warps; 2048 threads per SM | the launch (`choose_launch`) | occupancy, registers per thread, divergence |
| **Memory spaces** | RAM (31 GB in WSL), L3 30 MB; GPU 8 GB, L2, 64 KB constant | 24 GB HBM2 | `image.cu` (host), `gpu_image.cu` (device), the Scene in `__constant__` | the memory budget → passes; host ↔ device copies |

CPU facts: [Intel's specification](https://www.intel.com/content/www/us/en/products/sku/242292/intel-core-ultra-7-processor-255hx-30m-cache-up-to-5-20-ghz/specifications.html) (20 cores = 8 P + 12 E, 20 threads; boost 5.2 GHz P, 4.5 GHz E). Measured in WSL: 20 CPUs, 1 thread per core, L3 30 MB; GPU SM clock 825 MHz idle, 3090 MHz max.

## 3. One design at every level: the work queue

The same idea runs on both processors:

| | GPU | CPU |
|---|---|---|
| Workers | persistent threads, on the allowed SMs | one thread per allowed core |
| Taking work | `atomicAdd` on `next_chunk` (EP3's queue) | `atomic_fetch_add` on `next_chunk`: the same queue |
| Recording | `atomicAdd` on the image in device memory | atomic increment on the image in RAM |
| Random numbers | cuRAND, chunk = subsequence | `rng.cuh`, chunk = subsequence |
| Result | the same image whatever the number of SMs | **the same image whatever the number of cores** (integer adds commute) |

- **Why atomics on the CPU image too** (and not one image per thread, added at the end): memory stays one image whatever the cores, as on the GPU. Per-thread images would need 20 × 96 MB, and 20 × 1.3 GB at 1k. With 1 thread the atomic costs a little: to measure.
- **Decision B still holds:** separate types (`Image`, `GpuImage`), the physics shared; only the CPU's `count_photon` becomes atomic.

## 4. Benchmarking like HPC

The professor works in HPC; the results must survive the questions HPC asks. **Hoefler and Belli** (SC15, "Scientific Benchmarking of Parallel Computing Systems: Twelve Ways to Tell the Masses when Reporting Performance Results"), the constructive answer to **Bailey's** 1991 "Twelve Ways to Fool the Masses", found that 38% of papers reporting speedups did not state the base performance ([paper page](https://evaluate.inf.usi.ch/node/366.html), [slides](https://blogs.fau.de/hager/files/2017/02/hoefler-scientific-benchmarking-isc-pe-invited.pdf)). For EP4:

1. **State the baseline of every speedup.** GPU vs **one CPU core** and GPU vs **all 20 cores**: both, labelled. A GPU speedup over one core alone is one of Bailey's ways to fool the masses.
2. **Repeats and spread:** medians and their spread, not one run (already in the sweeps).
3. **Report the conditions:** clocks (the GPU goes from 825 to 3090 MHz), temperature, precision, build flags, versions (already in the records; clocks to add).
4. **Say what is hidden:** inside WSL the P- and E-cores look the same and Windows places the threads; the CPU curve may bend after 8 threads, and we cannot pin threads to P-cores from inside WSL. A result to show, not hide.

**The roofline** (Williams, Waterman and Patterson, CACM 2009: performance bounded by the peak compute or by memory bandwidth × operational intensity, [paper](https://www.osti.gov/servlets/purl/963540)) explains *why* a kernel runs at the speed it does. Without hardware counters we cannot place the kernel on it exactly; for Tuesday it is a concept to draw at the board, with the FP64 (compute-bound on a 1:64 GPU) vs FP32 (atomics) contrast, supported by our section timers (section 8).

## 5. The files, by level

| Level | Files |
|---|---|
| The pass loop (any processor) | `main.cu`, `options.cu`, `scene.cu`, `observers.cu`, `report.cu`, `files.cuh`, `run.cuh` |
| The physics (any processor) | `photon.cuh`, `real.cuh`, `geometry.cuh`, `rng.cuh` |
| The processor interface | **`processor.cuh`** |
| The CPU | **`cpu.cu`** (the processor), **`cpu_cores.cu`** (threads, the queue, one per core), `image.cu` (host memory) |
| The GPU | **`gpu.cu`** (the processor: constant Scene, the kernel, the launch), **`gpu_sms.cu`**, **`gpu_image.cu`**, **`gpu_device.cu`**, **`gpu_rng.cuh`** |
| The node | `engine.py` (build, run), later `machines/` and `remote` |
| The experiments | `sweeps.py`, `scaling.py` |

**Options and output, by level:**
- `--cores N` (CPU) and `--sms N` (GPU): the compute units allowed.
- Output: `PROCESSOR=CPU|GPU`, `PROCESSOR_NAME=…`, `COMPUTE_UNITS=` (allowed), `COMPUTE_UNITS_USED=`, `UNIT=core|sm`. `scaling.py` reads `COMPUTE_UNITS` and does not care which processor it was.
- The sweeps over compute units run on both processors: `strong` and `weak` over cores 1–20 and over SMs 1–36.

## 6. Today's scope, with this

| Step | Change from the plan |
|---|---|
| 1 | the interface is `processor.cuh`; the CPU one is `cpu.cu` (still one thread) |
| 2–5 | unchanged: the GPU (`gpu.cu`, `gpu_sms.cu`, `gpu_image.cu`, `gpu_device.cu`) |
| **5b (new)** | **`cpu_cores.cu`:** the CPU on N cores with the same queue; check: the same image for 1 and 20 cores |
| 7 | the sweeps over compute units on **both** processors (cores 1–20, SMs 1–36); `devices` states both baselines |
| 8 | `scaling.py` plots cores and SMs side by side |

Cost of 5b: one more file (~100 lines), and the CPU sweeps (~30 min of running). **If time runs short, 5b is the first thing to drop:** the GPU's story stands without it, and the CPU baseline is then one core, said so explicitly.

**Update 2026-11-21: 5b is deferred** ("we don't need to go much further on the CPU for now, but we want the same tests"). The CPU stays one thread, runs all the checks, and is the baseline of one core. Today goes to the GPU's map and observability instead (sections 7–8).

## 7. The code, mapped to where it runs (the GPU, one photon)

Which hardware runs each piece, and where its data lives. "Unit" is the part of the SM that does the work; the FP64/FP32 column says what changes with the precision. To be **confirmed per piece** by the instruction mix (section 8).

| # | Code | Data lives in | Unit | FP64 vs FP32 |
|---|---|---|---|---|
| 1 | `curand_init` once per chunk; `curand` per number (Philox: 10 rounds of 32-bit multiply-high and XOR) | registers (the state) | **integer units** (`IMAD`, `LOP3`, `IADD`) | the same in both: integers |
| 2 | the number → `real` in (0, 1] | registers | int → float conversion (`I2F`) + one multiply-add | `DFMA` vs `FFMA` |
| 3 | `sample_surface_point`: `cos`, `sin` of φ | registers | no hardware sine (we do not use `--use_fast_math`): a polynomial after range reduction | **FP64 units** (`DFMA`, 1:64 on the laptop) vs **FP32 cores** (`FFMA`) |
| 4 | `sine_from_cosine`, `sqrt` | registers | the **special function unit** (`MUFU`, a reciprocal square root) gives a first guess; multiply-adds refine it | FP64: an FP64 guess + several `DFMA`; FP32: `MUFU` + few `FFMA` |
| 5 | `sample_emission_mu`: `sqrt` or `cbrt` | registers | `cbrt` is all software | many `DFMA` vs `FFMA` |
| 6 | `escapes_spots`: the spot centers | **constant memory** (the Scene) | the constant cache: **one address for the whole warp, a broadcast**: as fast as a register | the chord test in `real` |
| 7 | `observer_bin`: `atan2`, then `slot_of_imu[imu]` | constant memory | `atan2` in software; the table read: **each thread a different `imu`, so the constant cache serializes the warp** (up to 32 reads) | `DFMA` vs `FFMA` + `MUFU` |
| 8 | `pixel_index` | registers | FP math, then integer index arithmetic | `real` |
| 9 | **`count_photon_atomic`** | **the image, in device memory** (GDDR7; 96 MB, bigger than L2) | **the atomic units in the L2 cache** (`RED.E.ADD`: an add with no reply; the thread does not wait) | the same: integers |
| 10 | the photon tallies | registers; at the end, one `atomicAdd` per counter per thread | L2 atomics | — |
| 11 | the queue, `take_next_chunk` | one counter in device memory | L2 atomics on **one hot address** (once per 1024 photons) | — |
| 12 | branches: blocked by a spot, missed the observers, recorded | — | **divergence**: lanes of a warp that take different paths run one after the other | — |

**Around the kernel (the host's side, per pass):**

| Code | Where |
|---|---|
| `load_pass`: the Scene → constant memory (`cudaMemcpyToSymbol`, ~18 KB) | over PCIe, into the GPU's constant bank |
| `zero_pass_image`: `cudaMemset` | a small kernel the driver launches on the GPU |
| `observer_on_host`: `cudaMemcpy` device → host | the GPU's **copy engine** (DMA), over PCIe |
| `save_counts`, the record | the CPU, the disk |

**What the map already predicts, to measure:**
- **FP64 vs FP32 on the laptop:** pieces 2–8 run on FP64 units at 1/64 of the FP32 rate, so the FP64 kernel should be limited by arithmetic; the FP32 kernel by something else (atomics, divergence, the constant-cache serialization). On the A30 (1:2) the gap should almost close.
- **Row 7 is a suspect:** a constant-memory table read with a different index per thread is the slow case of constant memory. **Experiment:** the same table in global memory (read through the L1 cache) vs constant memory; compare `PHOTONS_TIME`.
- **Row 9:** ~8% of the photons are recorded, so the atomics are a minority of the work, but each is a trip to L2.

## 8. Observability: what we measure, and how

| What | How | When | Where it shows |
|---|---|---|---|
| **Registers per thread, stack, spills, constant and shared memory per kernel** | the compiler's report (`nvcc -Xptxas -v`), kept by `engine.py` at build; and `cudaFuncGetAttributes` at run time | build; each run | `build/star_gpu64.report`; `REGISTERS_PER_THREAD`, `LOCAL_BYTES_PER_THREAD`, `CONSTANT_BYTES` in the output |
| **The instruction mix** (which units the code uses) | `cuobjdump -sass` of each binary, instructions counted by class: integer, FP64, FP32, special functions (`MUFU`), atomics (`RED`, `ATOM`), constant (`LDC`), global (`LDG`/`STG`), local = spills (`LDL`/`STL`), branches | build | `instruction_mix.py`: a table FP64 vs FP32 for the class |
| **Occupancy** | the occupancy API: blocks per SM, active warps / maximum warps per SM | each run | `BLOCKS_PER_SM`, `OCCUPANCY` |
| **The GPU's hardware** | device attributes: SMs, threads per SM, L2 size, memory clock and bus width (peak bandwidth), FP32:FP64 ratio, constant memory | each run | `CONFIG` / `SUMMARY` keys |
| **GPU memory use** | `cudaMemGetInfo` at start, after the context, after the image | each run | `GPU_MEMORY_TOTAL`, `GPU_FREE_BEFORE`, `GPU_FREE_AFTER` (the context's and the image's share) |
| **Work per SM** | counters indexed by `%smid` in the kernel: chunks taken, photons, recorded, cycles (slowest block) | each pass | one `SM` line per SM: **is the queue balancing the SMs?** |
| **Clocks, power, temperature, utilization, memory used, during the run** | `nvidia-smi` sampled every 0.5 s by Python next to the engine | each run of `star.py` and each sweep point | min / median / max per run, saved with the point |
| **GPU-side times** of the kernel, the zeroing and the copies | **CUDA events** (`cudaEventRecord` before and after each; the GPU stamps them itself) next to the host's clock | each pass | `KERNEL_TIME`, `MEMSET_TIME`, `COPY_GPU_TIME`: the GPU's own view vs the host's |
| **Time per section of the kernel** (how much, not only which) | **our own section timers:** a separate build (`-DSTAR_SECTION_TIMERS`, binary `star_gpu64_sections`) reads `clock64()` at the borders of each piece of the map (random numbers, surface point, direction, spots, bin, pixel, atomic) and adds the cycles per thread, then per SM | on request, and once per precision for the class | a `SECTION` line per piece: cycles and share of the kernel |

### No Nsight needed [decided 2026-11-21]

IAG's machine may not have Nsight, and the laptop's counters are blocked. So EP4 measures itself with what any CUDA install has: the runtime API (events, attributes, occupancy, memory info), `clock64()` in the kernel, `cuobjdump` and the compiler's report (shipped with `nvcc`), `nvidia-smi` (shipped with the driver).

- **The section timers replace `ncu`'s "how much":** the static instruction mix says *which* units the code uses; the timers say *how many cycles* each piece takes, per precision.
- **The observer effect:** reading the clock costs cycles and can change how the compiler schedules the code. So the timers live in their own build, never in the binaries we time for the sweeps, and the section build's total is compared with the normal build's `KERNEL_TIME` to show how much the measuring changed.
- **Nsight stays optional:** if `nsys` or `ncu` exist on a machine, they can confirm our numbers; nothing depends on them.
- **The CPU keeps the same checks** (split, passes, EP2, precision: all four on both binaries) and its phase times. Nothing more on the CPU for now: `cpu_cores.cu` (step 5b) is deferred, so CPU speedups are stated against **one core**, said so explicitly.

## 9. Open questions (research)

- [ ] The A30 node's CPU (cores, sockets, NUMA): only at IAG.
- [ ] Can WSL pin a thread to a P-core? (`sched_setaffinity` works on the 20 virtual CPUs, but they do not map to fixed physical cores.)
- [ ] Clocks during a run: sample the GPU's SM clock (`nvidia-smi`) and report it with each point.
- [ ] The roofline numbers for the laptop's GPU (peak FP32, FP64, memory bandwidth) from the datasheet, to draw at the board.

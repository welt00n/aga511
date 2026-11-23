# EP4 roadmap: where we are and how we got here

**The one place for the plan.** Phases are lettered (A, B, C…) so they don't get mixed up with the walkthrough's pieces (1–10) or the old step lists in [jobs_and_passes.md §9](jobs_and_passes.md#9-implementation-order) and [driver.md §9](driver.md#9-implementation-order), which now point here.

**The goal: Tuesday 2026-11-24** *(set 2026-11-21; the course's 12-01 date does not apply)*. EP4 is delivered **with the CUDA class: one 2-hour class**, the CUDA lesson together with the projects (EP3's Buffon and EP4's star). EP4 = EP2's spotted star rebuilt with CUDA, with performance tests (speedup, Amdahl, Gustafson).

## 0. Today: all of EP4 (Saturday 2026-11-21) [to review]

**Saturday: EP4, all of it. Sunday: prepare the class. Tuesday: the class.**

- **While building,** every CUDA (or HPC) concept we use goes into [../cuda_class/ep4_concepts.md](../cuda_class/ep4_concepts.md): what it is, where EP4 uses it, what we measured. Sunday's class prep starts from that file.
- **Each step ends with a check** against something we already trust (the CPU engine, EP2, the laws).
- **A short walkthrough after each step,** so every line can be explained in class.

| # | Step | What | Done when | Concepts for the class |
|---|---|---|---|---|
| 0 | **Commit** | the before-GPU changes | **done:** `0977989` | — |
| 1 | **F1: the processor split, then the GPU build** | first on the CPU: the photon loop moves from `main.cu` to `cpu.cu` behind `processor.cuh`; **the Makefile goes, `engine.py` builds** (one `nvcc` call per binary, rebuilt when the code hash changes); all checks unchanged. Then `star_gpu64` (and `star_gpu32`), SASS only, with `gpu_device.cu` (`PROCESSOR_NAME` = the GPU's name). Files: [gpu_engine.md](gpu_engine.md) | **done:** the CPU checks give the same numbers as before; `star_gpu64` builds (`sm_120` cubins only, no PTX) and prints `PROCESSOR=GPU`, its name, cc 12.0, 36 SMs | link-time choice of an implementation, what `nvcc` does (two passes, `__CUDA_ARCH__`), SASS vs PTX, compute capability |
| 2 | **F2: random numbers on the GPU** | cuRAND's `curand_init` + `curand` in a kernel, chunk = subsequence; `--test-rng` on the GPU | **done:** the raw numbers and the uniforms (FP64 and FP32) equal `rng.cuh`'s, bit for bit, for chunks 0, 1, 1 000 003 and 2³² + 5 | the cuRAND device API, counter-based RNGs, why our CPU copy exists |
| 3 | **F3: the star on the GPU** | `gpu.cu`: the Scene in `__constant__` (`cudaMemcpyToSymbol`, per pass); `GpuImage` with `atomicAdd`; the kernel `trace_photons` with EP3's persistent threads and work queue, launch size from the occupancy API; each thread's tallies added once at the end; one `cudaMemcpy` per observer into a host buffer of one observer (`COPY_TIME`); the same files and records. `star.py --compare-gpu` | **done: GPU = CPU** bit for bit in FP64 (10⁷ and 10⁸); FP32: 3.2·10⁻⁵ of the photons move (3/4 from FMA); split, passes bit for bit on the GPU; EP2 χ²/dof 0.9957; 75× one CPU core in FP64 | kernels, grids, blocks, warps, memory spaces, constant memory, atomics, the queue, occupancy, `cudaMemcpy`, synchronization |
| 4 | **F4: the SM mask, and work per SM** | `gpu_sms.cu` (`%smid`, as EP3), cooperative launch; `--sms n`; per-SM counters (chunks, photons, recorded, cycles) → one `SM` line per SM | **done:** the same image on 1, 2, 9, 36 SMs; chunks per SM within 0.1%; **the clock drops from 2.88 GHz (1 SM) to 1.95 GHz (36 SMs)** | SMs, `%smid`, why the blocks must all be resident, load balance |
| 5 | **F5: memory, passes and the kernel's resources** | `cudaMemGetInfo` (before, after the context, after the image) → `--memory-budget` default; passes (the loop stays on the host); `cudaFuncGetAttributes` (registers, local memory = spills, constant bytes), occupancy, the GPU's attributes (L2, bandwidth, FP32:FP64); build and driver versions | **done:** passes bit for bit on the GPU; **occupancy 0.50 (FP64, 80 registers) / 0.58 (FP32, 67): registers limit it**; passes chosen from free memory (10 GB job → 4 + 4); copy ~24 GB/s; at 1k the disk (save) is the serial part | device memory, registers and occupancy, host ↔ device traffic |
| 5b | ~~The CPU's cores~~ **deferred** | the CPU stays one thread; it runs all the checks and is the one-core baseline, said so ([hpc_architecture.md §8](hpc_architecture.md#8-observability-what-we-measure-and-how)) | — | — |
| 6 | **The map and the speed** | **the code mapped to the hardware** ([hpc_architecture.md §7](hpc_architecture.md#7-the-code-mapped-to-where-it-runs-the-gpu-one-photon)): the compiler's report and `instruction_mix.py` (FP64 vs FP32: integer, FP64, FP32, special functions, atomics, constant, spills); `nvidia-smi` sampled during runs (clocks, power, temperature, memory); CPU vs GPU, FP64 vs FP32 at a fixed N; **the `slot_of_imu` experiment** (constant vs global memory); **the occupancy experiment** (cap the registers with `-maxrregcount` / `__launch_bounds__`: more threads per SM vs spills); CUDA events for the GPU-side times; **our own section timers** (`star_gpu64_sections`, cycles per piece of the photon). No Nsight needed | **done:** arithmetic-bound (direction, surface, `atan2`); constant vs global memory: no difference; more occupancy: slower; FMA +7% in FP64; FP64 17× FP32 per photon; observer effect 0% (FP64) / 8% (FP32) | where each line runs, FP64 units vs FP32 cores vs SFU vs L2 atomics, constant-cache broadcast vs serialization, divergence, clocks, measuring without a profiler, the observer effect |
| 7 | **G1: the sweeps** | `sweeps.py`, defined in [gpu_engine.md §7](gpu_engine.md#7-the-sweeps-today-to-review): `strong` and `weak` over SMs 1–36 (FP64 and FP32), `problem-size`, `devices` (the GPU against one CPU core, stated); N calibrated; 3 repeats; ~1 h in the background | **done:** 7 sweeps, 198 points, no failure | strong vs weak scaling, p(N) |
| 8 | **G2: the scaling analysis** | Amdahl fit (p), Gustafson fit (s′), efficiency, strong and weak side by side, the serial part measured (SETUP + ZERO + SAVE) vs fitted | **done:** strong FP64 35.9× (p = 0.9998); FP32 36.0× in cycles, 33.1× in seconds; weak flat, Gustafson s′ 0.054 vs 0.058 measured; Amdahl's ceiling 2.3× at N = 10⁸ | Amdahl, Gustafson, serial fraction |
| 9 | **H: the showcase** | one big run (size, frames and N set from the measured speed and the SNR formula), gifs (and mp4 if `ffmpeg` is there); pictures made one observer at a time | **done:** 1k, 300 frames, N = 3.9·10¹² in 385 s; SNR measured = predicted (14.1 / 11.7); 4 mp4s | SNR ↔ N, photons per pixel, μ bins |
| 10 | **Wrap up** | EP4 README results, the roadmap's log, commit | committed | — |

### Sunday 2026-11-22: the open questions, then wrap up EP4 [set 2026-11-22]

**Replanned:** Sunday = finish EP4, then prepare the class; Monday = fix and practise the class, the TCD, and the IAG runs (the A30 is reachable only from inside IAG, so phase E moves to Monday). EP4 asks for Amdahl's law; we already did more, so nothing new is built: only the open questions and the issues found.

| # | Question | Hypothesis | How we check | Result |
|---|---|---|---|---|
| Q1 | 4 FP64 instructions in the FP32 kernel | a `double` literal or a math overload promoted to double | `ptxas --warn-on-double-precision-use`; the SASS around them | **closed:** `sinf`/`cosf`'s slow argument reduction (Payne–Hanek), for \|x\| > ~10⁵ only: never runs for our angles |
| Q2 | 1063 `NOP`s in the FP64 kernel | alignment padding, almost never executed (the mix is static) | where they sit in the SASS | **closed, guess wrong:** runs of 4 between dependent FP64 instructions: the warp waits for the slow FP64 unit; they are executed |
| Q3 | pass 2 ran 5× slower after a 12 s save | the GPU's clock dropped while it idled | cycles per photon (same → the clock) and `nvidia-smi` across a long save | **closed, guess wrong:** not reproduced; a pass after the clock drop was the fastest; ~30 ms kernels vary ±30%, ~170 ms ones 1% |
| Q4 | the CPU was slower during the sweeps | the CPU and the GPU share the laptop's power | the CPU alone vs beside a GPU run | **closed, guess not supported:** 8.8–9.1·10⁶ in the morning, 7.6–9.4·10⁶ by core in the afternoon (`research/cpu_baseline.py`): the CPU's own ~20% spread covers the sweep's 7.0–8.0·10⁶ |
| I1 | 4 `observer_*` files in `EP4/` | a manual run without `-o` writes into the current folder | the records (`CODE_HASH=none`, 1 SM, 20:26) | **fixed:** the 4 records had been committed in `3a230c7`; deleted, and `/EP4/observer_*` ignored |

### Research to keep: standalone scripts that reproduce each investigation [done 2026-11-22]

Not a product: one script per question in `EP4/research/`, each saving its output in `research/results/`; the log with the question, the method, the result and its meaning is [research.md](research.md). **From now on every investigation gets its script and its entry there when it's done**, not later.

| Investigation | What it shows | Script |
|---|---|---|
| FP64 instructions in the FP32 kernel (Q1) | `sinf`/`cosf`'s slow argument reduction, never run | `fp64_in_fp32_kernel.py` |
| The `NOP`s (Q2) | runs of 4 between dependent FP64 instructions | `nop_runs.py` |
| Pass 2 slower after a long save (Q3) | not the idle clock; short kernels are noisy | `pass_order.py` |
| The CPU baseline between sessions (Q4) | 8.8–9.1·10⁶ today vs 7.0–8.0·10⁶ in the sweep; cause open | `cpu_baseline.py` |
| Why the clock falls | `nvidia-smi` clock reasons: power cap in FP32 on 36 SMs, voltage ceiling otherwise | `clock_reasons.py` |
| The SM's limits | `cudaGetDeviceProperties`: 65 536 registers, 48 warps, 24 blocks per SM | `sm_limits.py` |
| Register sweep | occupancy follows the registers; spills cost up to 9% (FP64); now interleaved, in cycles too | `register_sweep.py` |
| Registers on the A30 (compile only) | 96 / 76 registers on sm_80 → occupancy 0.33 / 0.39 | `a30_registers.py` |
| Why sm_80 needs more registers | the front end: compute_≤90 vs compute_≥100 (CUDA 13's newer NVVM) | `front_end_registers.py` |
| Where the registers go (liveness) | the peak of live registers is `sin`/`cos` of the local azimuth (photon.cuh:56–57) | `register_liveness.py` |
| (a tool) bit-for-bit reference | `save` before a refactor, `compare` after | `bit_for_bit_reference.py` |

**On Monday at IAG:** `sm_limits.py`, `register_sweep.py`, `clock_reasons.py` and `register_liveness.py` on the A30 answer whether more occupancy helps there.

### Next, in order [2026-11-22]
1. ~~The research scripts above, and their research log.~~ Done.
2. ~~The products~~ **built 2026-11-22** ([walkthrough piece 25](walkthrough.md#piece-25-the-products-mainpy-lab)): `test`, `make`, `measure`, `compare` in one `main.py`; the preflight (environment, prediction, confirmation); the minimum N ([gpu_engine.md §10](gpu_engine.md#10-valid-runs-the-minimum-n-decided-2026-11-22)); the report as an executed notebook and a PDF. Then: the full laptop `measure`, and its numbers into the docs.
3. **Monday at IAG:** `test`, `measure`, then `compare` with the laptop.
**A note on "GPU = CPU" (step 3).** The random numbers will be identical (same Philox). The arithmetic may not be: the GPU's `sin`, `cos`, `atan2`, `cbrt` come from CUDA's math library, not glibc, and `nvcc` fuses multiply-adds (FMA) where `gcc` does not. Both change the last bit of some results, and a photon right on a pixel edge (or a spot edge) can then land on the other side. So:
- expect the **counts identical or nearly**, and the images identical or differing by a few photons;
- if they differ, count the differing photons, and try `--fmad=false` to see how much comes from FMA;
- either way it is a result for the class: **bit-for-bit vs statistically equal**.

**Order and fallbacks.** Steps 1–3 are the core: with them, EP4 runs on the GPU. If time runs short, step 4's SM mask is what the Amdahl sweep needs, so it comes before step 5; step 9 can be smaller.

### Deferred (not for Tuesday)
- **D in full** (jobs, resuming, backups, prompts, TOML): the sweep script is enough.
- **E and the preflight** (IAG): the A30 stays a prediction in the class ([gpus.md](gpus.md)).
- Progress lines while the GPU runs, presets, `np.memmap` analysis, Nsight Compute.
- **The monograph,** if it is required apart from the class (to confirm).

---

## 1. The phases

| Phase | What | Status | Docs |
|---|---|---|---|
| **A** | Understand EP3 (Buffon on the GPU) and CUDA's basics | done; it keeps feeding the class | [README.md](README.md), [../cuda_class/](../cuda_class/class_plan.md) |
| **B** | The sequential engine, written in the shape the GPU needs: Philox, chunks, a pure photon, the Scene, FP64/FP32 | **done** 2026-11-20 | [walkthrough.md](walkthrough.md), [precision.md](precision.md), [cpp_for_python.md](cpp_for_python.md) |
| **C** | The engine with passes: observers in radians, K per pass or a memory budget, one file and one record per observer, progress lines | **done** 2026-11-21 | [jobs_and_passes.md §15](jobs_and_passes.md#15-implemented-the-engine-with-passes-cpu-2026-11-21) |
| **D** | The local driver: job files and prompts, the job folder, fingerprint, resuming, backups, `plan`, `run`, `status`, `stop`, `clean` | designed | [driver.md](driver.md), [jobs_and_passes.md](jobs_and_passes.md) |
| **E** | Remote runs (IAG A30) and machine analysis: `remote setup/run/status/fetch`, `probe`, `calibrate` | designed | [driver.md §2, §4](driver.md#remote-remote-command-host), [gpus.md](gpus.md) |
| **F** | **The GPU engine:** kernels for the chunk loop, cuRAND on the device (bit for bit against the CPU copy), the work queue, the image with `atomicAdd`, the Scene in constant memory, the SM mask, passes on the device | not started | [README.md](README.md), [gpus.md](gpus.md) |
| **G** | Sweeps and the scaling analysis: Amdahl (strong) and Gustafson (weak), over SMs and image sizes, laptop vs A30 | designed | [driver.md §6–7](driver.md#6-sweeps-the-runs-of-a-scaling-study), [scaling_laws.md](scaling_laws.md) |
| **H** | The showcase: presets (512, 1k, 2k, 4k), video (mp4), the big runs at IAG | designed | [jobs_and_passes.md](jobs_and_passes.md), [gpus.md](gpus.md#how-big-can-the-image-be) |
| **I** | The monograph and the presentation | not started | — |

**The order:** F → G (minimal) → the class, set by the Tuesday deadline (section 0). D and E are reduced to a sweep script; the rest is deferred.

### How we work now [decided 2026-11-21]

1. **Walkthrough of phase C first** ([walkthrough.md](walkthrough.md), pieces 10–15), so the passes code is understood before anything is built on it.
2. **Every task found on the way goes into the backlog below,** with the phase it belongs to.
3. **Then implement the backlog together:** both D (the driver) and F (the GPU engine), in an order settled at that point ("The order to decide", above).

---

## 2. How we got here (the story)

1. **Understanding EP3 first** (phase A). Before writing EP4: Buffon on the GPU, line by line. Host and device, the thread hierarchy, warps, SMs, persistent threads, the work queue with `atomicAdd`, SM masking, Philox streams, why Amdahl's p grows with N. → the concept list in [README.md](README.md).
2. **Deciding how to build EP4** (phase B). Sequential first, but in the GPU's shape:
   - our own copy of cuRAND's Philox, so the CPU draws exactly what the GPU will draw;
   - a fixed chunk size with one random stream per chunk, so any split of the work gives the same image;
   - `trace_photon` pure (it returns a pixel and writes nothing);
   - the `real` switch for FP32.
3. **Checking it** (phase B): Philox against the Random123 test vectors; two halves of the chunk range = the full run, bit for bit; EP4 vs EP2, χ²/dof = 1.0047; FP32 vs FP64. First commit.
4. **Docs and the class.** `docs/` for every concept; `cuda_class/` with the class plan; Buffon profiled with Nsight Systems (its serial part is CUDA context creation). Nsight Compute is waiting for the GPU counters to be enabled.
5. **Precision, in depth.** What 53 bits mean, how FP32 changes the physics, `sine_from_cosine`, the FP32 hotspot at pole-on observers. → [precision.md](precision.md). Commit.
6. **C++ for a Python programmer.** Headers with whole functions, `static inline`, include guards, templates with `HD`. → [cpp_for_python.md](cpp_for_python.md). Commit.
7. **The walkthrough, pieces 1–9:** the engine read piece by piece, with your questions. Along the way: the recording decision (B: `count_photon` on the CPU, `atomicAdd` on the GPU, the physics shared). → [walkthrough.md](walkthrough.md).
8. **Piece 9 opened a bigger question:** "how big can the image be?" The goal of 1080p at 30 fps led to memory limits per GPU, the laptop vs the A30, square sizes (1k, 2k), and Gustafson's law. → [gpus.md](gpus.md), [scaling_laws.md](scaling_laws.md).
9. **Memory became passes.** "What if one observer needs 7 GB?" → record K observers per pass, re-simulating the same photons each time. From there: jobs that can be resumed, fingerprints, backups, the code hash, SSH to IAG, N ↔ SNR, nmu from the frames, machine analysis. → [jobs_and_passes.md](jobs_and_passes.md).
10. **The driver's design.** Commands, TOML files and prompts, remote as an explicit and light command group, the engine writing its own records, sweeps for Amdahl and Gustafson. → [driver.md](driver.md).
11. **Phase C implemented** (today), with three findings (the log below).

---

## 3. Log: decisions and findings

| Date | What | Where |
|---|---|---|
| 2026-11-20 | Own copy of cuRAND's Philox on the CPU (cuRAND's device API cannot run on the host) | [README.md](README.md), [walkthrough.md piece 3](walkthrough.md#piece-3-random-numbers-philox-rngcuh) |
| 2026-11-20 | Chunk size fixed at 1024, one Philox subsequence per chunk; spots from a reserved stream | [walkthrough.md](walkthrough.md) |
| 2026-11-20 | First checks: split bit for bit; χ²/dof = 1.0047 vs EP2; FP64 1.0·10⁷ photons/s; FP32 1.58× faster, 5·10⁻⁴ photons moved | [../EP4/README.md](../EP4/README.md) |
| 2026-11-21 | Buffon's serial part is CUDA context creation (~0.15 s), measured with `nsys` | [../cuda_class/nsight.md](../cuda_class/nsight.md) |
| 2026-11-21 | FP32's real hotspot: the projection for pole-on observers (divides by sin i) | [precision.md](precision.md) |
| 2026-11-21 | Recording decision B: shared physics; `Image` + `count_photon` on the CPU, `DeviceImage` + `atomicAdd` on the GPU | [README.md §3](README.md) |
| 2026-11-21 | Passes, idempotent jobs, records, SNR ↔ N, nmu from the frames | [jobs_and_passes.md](jobs_and_passes.md) |
| 2026-11-21 | Driver: TOML and prompts; explicit `remote` (no Python on the server); the engine writes its records; `--detach` | [driver.md](driver.md) |
| 2026-11-21 | **Phase C done:** passes bit for bit; χ²/dof = 0.9957 vs EP2 | [jobs_and_passes.md §15](jobs_and_passes.md#15-implemented-the-engine-with-passes-cpu-2026-11-21) |
| 2026-11-21 | **Finding:** with an even `nmu` the equator is a bin edge → `nmu` is now the odd number nearest frames/π | [jobs_and_passes.md §11](jobs_and_passes.md#11-frames-and-μ-bins-nmu-follows-from-the-frames) |
| 2026-11-21 | **Finding (corrected):** a pass costs a full trace; writing to the image costs ~180 ns per recorded photon; no cache effect visible. My first explanation (cache) was wrong | [jobs_and_passes.md §15](jobs_and_passes.md#what-we-learned) |
| 2026-11-21 | Walkthrough of phase C (pieces 10–15): 9 engine changes to make before the GPU | [walkthrough.md recap](walkthrough.md#recap-of-pieces-1015-what-we-learned), [backlog C′](#c-before-the-gpu-engine-changes-the-gpu-will-inherit) |
| 2026-11-21 | `fsync` before the rename: measured cost 0.15–0.18 s per 315 MB vs 0.08–0.16 s without | [walkthrough.md piece 13](walkthrough.md#questions-on-piece-13) |
| 2026-11-21 | **Preflight checks**, local and remote: can this machine run the job, before running it | [backlog](#preflight-can-we-run-here-local-and-remote-asked-2026-11-21) |
| 2026-11-21 | **Deadline: Tuesday 2026-11-24,** EP4 delivered with a 2-hour class (CUDA + the projects). Plan replanned: F → G minimal → the class | [section 0](#0-today-all-of-ep4-saturday-2026-11-21-to-review) |
| 2026-11-21 | **EP3 review (its sessions, from the `ai-memory` store):** the queue, the launch from the occupancy API, the SM mask and the cooperative launch were tested there and are reused as is; `CYCLES` and `REGISTERS_PER_THREAD` added; EP3's adaptive chunk size deliberately not taken | [gpu_engine.md §8](gpu_engine.md#8-what-ep3-already-settled-review-of-the-ep3-sessions-2026-11-21) |
| 2026-11-21 | **The sweeps defined:** a point = one engine process; `strong` and `weak` over 1–36 SMs (FP64 and FP32), `problem-size` (p(N)), `devices`; N calibrated so the 36-SM run lasts 5 s; 3 repeats; ~1 h in total | [gpu_engine.md §7](gpu_engine.md#7-the-sweeps-today-to-review) |
| 2026-11-23 | **The class around the CUDA Programming Guide, with a full script.** The guide (v13.5) read and summarized against our deck and code ([cuda_guide_reference.md](../cuda_class/cuda_guide_reference.md)); it agrees with the deck except on occupancy, where our measurement contradicts its default advice (a teaching point). The deck rebuilt as 10 parts with four "Guia" parts using our code (95 slides), built from `parts/` and `notes/` by `build_deck.py`; every slide with cues, a full script and actions (`roteiro.html` to study); a tablet presenter page; the handout `guia.html`. Measured for the memory part: the star in global instead of constant memory changes nothing (0.996× cycles FP64, 1.007× FP32) | [class_plan.md](../cuda_class/class_plan.md) |
| 2026-11-23 | **The night's runs, both done.** The 2k image (`make --size 2048 --snr 10`, the A30's 4 observers): N = 1.56·10¹³, 5590 s of photons at 1.1·10¹⁰/s, 4 passes, 1 h 43 min in all (predicted 1.6 h); SNR center 14.0 (14.1 predicted), limb 11.7 (11.7). The 1k ladder (`make --ladder 1e9:1e13 --size 1024`): 5 rungs, SNR at the limb 0.19 … 18.67 against 0.19 … 18.77 predicted; the last rung 31 min; the spots show from 10¹⁰. Both in the deck (the ladder slide and a new 2k slide) | `runs/make/2026-11-23_02-19-28-2048-…`, `runs/make/2026-11-23_04-02-09-ladder-1024-…` |
| 2026-11-23 | **The class, rebuilt around the orchestrator.** Rule: every point in the class is obtained by `main.py`. The map, point → command → output: [class_points.md](../cuda_class/class_points.md). Two gaps closed in `measure`: **the phases of a run** (setup, zeroing, photons, copy back, save; `strong-fp*.phases`, `figures/phases_fp*.png`, report 5.1: at 1k FP64 the photons are 97% of a 1-SM run and 52% of a 36-SM run, saving 5 GB ~4.3 s either way) and **instructions per register cap** (`occupancy.rows.*.instructions_vs_uncapped`, local stores/loads, report 4: FP32 at 55 registers with 0 B spilled has 1.047× the instructions for 1.011× the cycles). The 1k laptop measurement re-analysed with `measure --continue` (0 of 178 runs to do; report and figures rebuilt). The deck: 77 slides, 11 parts, ~25 SVG diagrams, a new part on random numbers (LCG vs Philox, a Philox round, a stream per chunk, six levels of validation with the real `--test-rng` output). **Overnight:** `make --size 2048 --snr 10` with the A30's 4 observers (1.6 h predicted) and `make --ladder 1e9:1e13 --size 1024` (~35 min), log in `EP4/runs/night_2026-11-23.log` | [class_points.md](../cuda_class/class_points.md) |
| 2026-11-23 | **Amdahl's ceiling figure on linear axes, with the cycles and the clock beside it** (for the class). The clock is now taken during the photons (the slowest SM's cycles ÷ the photon time). The `nvidia-smi` median over the whole process also counts the ~5 s of zeroing and saving: at 36 SMs it said 2610 MHz (FP64) and 2017 MHz (FP32), while the photons actually ran at 2878 and 2730 MHz. So FP64 holds its clock on all SMs (36.3× in seconds, 36.1× in cycles), and FP32 loses 6% to the power cap (35.3× vs 36.3×). The FP32 photon fit gives p = 1.0049 > 1 (the 1-SM run ran 3% slower in clock): it is no longer extended beyond the data. `strong_fp*` and the report table use the new clock; the 1k run's ceiling figures were redrawn from `parts/scaling.json` (its `strong_fp*.png` and report keep the old clock until re-analysed) | — |
| 2026-11-23 | **The slides move to HTML (reveal.js), with the phone as the remote.** The Beamer deck played its videos only in Acrobat; Microsoft 365 isn't free for the user and Google Slides needs internet for video. `cuda_class/deck/`: the same 33 slides in reveal.js 5.1.0 with KaTeX 0.16.11 (both from the npm registry, checksums matching, kept in `lib/` so nothing loads from the internet); the three videos (around, grid, SNR ladder) cut to one rotation and looped by the browser; step-by-step highlighting in the kernels; speaker notes (S). `apresentar.py` (standard library only; runs on the laptop's Windows Python 3.8, since WSL's network isn't reachable from the phone) serves the deck and a phone page with next/back, black screen, the slide's notes and a 120-min timer, over the phone's hotspot. Tested in the browser: the phone's next moves the deck and the notes follow; videos are served with byte ranges | [deck README](../cuda_class/deck/README.md) |
| 2026-11-23 | **Class planning starts; the A30 predicted.** The professor's syllabus transcribed and the class sequence drafted ([class_plan.md](../cuda_class/class_plan.md)); EP3 stays as delivered (Amdahl only). The A30 predicted from the laptop's photons per cycle per SM and the datasheet: FP32 4.3–8.6·10⁹/s, FP64 2.6–7.3·10⁹/s (the precision trade-off flips); the 2k showcase 31–61 min. `EP4/environment.yml` for setting up at IAG (conda-forge only) | [research.md](research.md) |
| 2026-11-22 | **No small N, ever:** every timed run in `measure` is above the minimum N. Removed the small-N strong sweep (10⁸) and the bottom of p(N) (10⁵…10⁹) and the "below the minimum by design" label; p(N) runs at 0.4×, 1×, 2× the reference N; the image fingerprints at 2·10¹⁰ (GPU only). Amdahl's ceiling is shown at valid N: on 36 SMs a 1k run is 6.0 s of photons + 5.6 s of setup, copy and saving 5 GB, so the whole run is 19× where the photons are 36× (ceiling 1/s ≈ 40). The validation checks against the CPU stay at 10⁸: they compare images, not times | [walkthrough piece 25](walkthrough.md#no-small-n) |
| 2026-11-22 | **Runs cleaned up (with the user's yes):** deleted the two earlier 4-observer 1k showcases (1.15 GB, replaced by the 7-observer one) and `~/aga511-old-runs` (4.9 GB, the runs from before the restructure); the test runs are kept. `runs/` is 2.4 GB, mostly mp4s (ignored by git) | — |
| 2026-11-22 | **The 1k measurement** (1024² × 300 frames × 4 observers, 5 GB per run): 227 runs in 89 min (predicted 88). Photon phase 36.3× (FP64) / 35.3× (FP32) on 36 SMs vs one SM, but the **whole run only 19.3× / 16.3×** (33.7× / 31.4× at 200²): on all SMs a run is 6.0 s of photons and 5.6 s of zeroing, copying and saving 5 GB. Weak scaling: serial share 0.48 on 36 SMs. Small N: 1.07×. **Fixed on the way:** the checks loaded whole 1k images into RAM and the kernel killed Python at 22 GB; they now stream each observer a few frames at a time (peak 4.9 GB), and the χ² compares 8×8-pixel blocks (at 1k a pixel holds ~1 photon at N = 10⁸) | [EP4 README](../EP4/README.md), `runs/measure/2026-11-22_20-25-24-Welton-all/` |
| 2026-11-22 | **Resolution vs image size, the two axes of cost** (for the class): `measure` gets a `costs` part and report section 6 (`lab/budget.py`): the bins (frames, nμ) set the blur and the photons, N ∝ nμ·frames·size²; the image (observers × frames × size² × 4 B) sets the memory and the passes. At 1k/300 frames the blur is ~11 px (equator), up to 105 px in the polar bin; a sharp 2k (600 frames) needs 16× the photons and doesn't fit this GPU. `research/binning_costs.py` | [research.md](research.md) |
| 2026-11-22 | **Every GPU product prints the GPU and the launch:** SMs, threads and warps per SM, registers per SM; per precision, registers per thread → threads per block × blocks per SM = warps per SM, occupancy, threads in all (laptop FP64: 80 registers allow 25 warps, 24 launched, as blocks fit only whole). On the A30 it will show its own | — |
| 2026-11-22 | **The 7-observer 1k showcase** (`make --snr 10`): 2 passes, 770 s (predicted 13 min), SNR measured = predicted (14.1 / 11.7); the southern views show no spots: all 3 spots are in the northern hemisphere. **The ladder, tested** at 256² (10⁸…10¹¹): SNR at the limb 0.23, 0.75, 2.36, 7.47 against 0.24, 0.75, 2.38, 7.51 predicted; the spots appear between 10⁹ and 10¹⁰. The full 1k measurement started (227 runs, 89 min predicted) | `runs/make/2026-11-22_20-07-02-…`, `runs/make/2026-11-22_20-23-46-ladder-…` |
| 2026-11-22 | **High resolution is the goal: 1k minimum for measurements, 2k maximum.** `measure` now traces the showcase's image (1024² × 300 frames × 4 observers, 5 GB per run, `--size` 1024–2048; the first full measurement used 200² × 150 frames); `make` accepts up to 2048 (2k on the A30). **The SNR ladder:** `make --ladder 1e9:1e12` runs one decade of N at a time (the same seed: each smaller run is exactly the first chunks of the larger, so the rungs are nested like partial sums) and makes a ladder video, the observer nearest 60° at every N side by side | [README](../EP4/README.md) |
| 2026-11-22 | **7 observers in `make`, both hemispheres** (0°…180° every 30°; `measure` keeps 4). **Fixed:** the plan's time ignored the passes (each pass traces all N again); the 7-observer 1k run needs 2 passes, ~13 min, not 6 | [walkthrough piece 25](walkthrough.md#the-videos) |
| 2026-11-22 | **The videos:** one rotation is exactly `--frames` frames (checked: the last frame is one step before the first); 10 s looked short. `make --loops N` (default 5) repeats each video without re-encoding (`ffmpeg -stream_loop`). Two new videos with all the observers: `observers_grid.mp4` (side by side, labelled) and `observers_around.mp4` (each view placed around a drawing of the star at its inclination from the rotation axis). `showcase.py` split into `snr.py` and `videos.py` | [walkthrough piece 25](walkthrough.md#the-videos) |
| 2026-11-22 | **No more comparison with EP2: the reference is EP4's sequential CPU.** The EP2 check (compiling and running EP2's own C program) is gone from the code (`checks`, `analysis`, `engine`, `speeds`, `test`, `measure`, the report). New check: the GPU vs the sequential CPU with **independent photons of the same star**: the same seed, chunks [0, C) on the CPU and [C, 2C) on the GPU. A first version used different seeds and failed at 10⁸ (χ²/dof 1.021, ~5σ): the seed also places the spots, so another seed is another star. Now χ²/dof 1.0035 (+0.9σ) at 10⁸; 0.998 in `test` | [walkthrough piece 25](walkthrough.md#the-map-from-the-old-names) |
| 2026-11-22 | **The "70× one core" headline is the classic unfair comparison** (a whole GPU vs one core of an untuned sequential program; Lee et al., ISCA 2010). Like for like from the same measurement: one SM vs one core 2.0× (FP64), 23.6× (FP32); all SMs vs a 20-core bound (one core's speed × 20) ≥ 3.5× (FP64), ≥ 39× (FP32). **Decided (2026-11-22): no speed comparison with the CPU at all.** The unit is the SM: `measure`'s `devices` part became `speed` (one SM and all SMs, the same N; 35.2× FP64, 32.9× FP32 on 36 SMs); the GPU = CPU checks report correctness only; `compare` shows one SM instead of a CPU core. EP4 vs EP2 is for the TCD | [walkthrough piece 25](walkthrough.md#the-unit-of-speed-is-the-sm-not-the-cpu) |
| 2026-11-22 | **Videos are mp4 only:** the gif fallback is gone; `make` requires `ffmpeg` (the environment check fails without it) and stops if `ffmpeg` fails. The media size is now predicted from the pixels (~0.15 bytes per pixel per frame: 46 MB per 1024², 300-frame observer) | [walkthrough piece 25](walkthrough.md#the-map-from-the-old-names) |
| 2026-11-22 | **The full laptop measurement** (`main.py measure`, all parts, valid N, 3 repeats): 233 runs in 67.0 min, predicted 69.8 (−4%). GPU 6.4·10⁸ photons/s FP64 (70× one core), 1.06·10¹⁰ FP32 (777×); strong 35.2× FP64 (p = 1.0000), FP32 32.9× in seconds / 35.9× in cycles; weak FP64 s′ 0.055; small N ceiling 2.3×; registers: FP64 flat to 64 (72 is 0.8% fewer cycles), FP32 fastest uncapped; validation passed. **What moved from 2026-11-21:** FP64 5.9 → 6.4·10⁸ (valid N now; the old devices point was 10⁸, below the minimum); 82× → 70× one core (the CPU was slower in that session, Q4); strong FP64 35.9× → 35.2×; weak FP64 measured s′ 0.058 → 0.041 (larger N per SM); p(N) at 10⁵ 0.14 → 0.006 (noise at a 0.4 s run). README results refreshed | [EP4 README](../EP4/README.md#results-on-the-gpu-2026-11-21-rtx-5070-laptop-gpu-36-sms), `runs/measure/2026-11-22_17-08-40-Welton-all/` |
| 2026-11-22 | **The products, built:** `main.py` with `test` (12 checks, ~20 s, all pass), `make` (N from the SNR, mp4 + stills + SNR check), `measure` (validation, devices, kernel, occupancy, scaling; 233 engine runs, 12 shared, ~70 min predicted; resumable with `--continue`) and `compare` (speeds, per SM per GHz, scaling, occupancy, image digests). Every product checks the environment, predicts time/memory/passes/disk and asks. **Minimum N** = max(1 s × speed, 32 chunks × resident threads × 1024): laptop FP64 9.1·10⁸, FP32 1.1·10¹⁰, CPU 8·10⁶. The report is a notebook executed by nbclient and a PDF through nbconvert + tectonic (installed from conda-forge with pandoc; pandoc 3's `\LTcaptype{none}` needed a `\newcounter{none}` for tectonic's older LaTeX). Tested on quick, explicitly invalid runs; two laptop measurements compared: identical image digests, speeds within 5%, and the GPU's FP64 digest equals the CPU's. `kernel_experiments.py` and `sweeps.py` removed (their work is points now) | [walkthrough.md piece 25](walkthrough.md#piece-25-the-products-mainpy-lab) |
| 2026-11-22 | **The research kept as scripts:** `EP4/research/`, one standalone script per investigation (Q1–Q4, clock reasons, SM limits, register sweep, A30 registers, the front end, liveness) plus `bit_for_bit_reference.py`; each saves its output in `research/results/` and ends with what we found first. All re-run: Q1–Q3, the clock, the limits, the A30 and the front end reproduced. **Two new results:** the interleaved register sweep is clean in FP32 too (fastest uncapped; more occupancy costs 1–5% in cycles, even without spills; FP64 flat up to 0.67, then spills cost up to 10%); and Q4's re-run gave 7.6–9.4·10⁶ by core, so the CPU's own ~20% spread covers the sweep's gap (Q4 closed: no GPU effect) | [research.md](research.md) |
| 2026-11-22 | **Why the A30 build needs more registers (96 vs 80 in FP64, 76 vs 67 in FP32): the compiler's front end, not the hardware.** The same source compiled to PTX for compute_75…90 needs 96 registers (FP64); for compute_100…120, 80–84. The jump is exactly at compute_100, where CUDA 13 switches to its newer NVVM IR (LLVM 18; NVIDIA's release notes). The newer PTX has fewer branches (107 vs 90) and more selects. The back end (ptxas) adds a few: the same compute_80 PTX takes 96 on sm_80 and 98 on sm_120. The opt-in `-nvvm-latest` for older targets is a libNVVM/nvJitLink option; nvcc rejects it. The A30's predicted occupancy: 0.33 (FP64), 0.39 (FP32), depending on IAG's toolkit | — |
| 2026-11-22 | **Register sweep (laptop):** occupancy follows the register limit at every cap; FP64 gains nothing from 0.50 → 0.67 and loses up to 9% once registers spill (clock steady at 2880 MHz); FP32 inconclusive (the clock fell 2748 → 2017 MHz during the sweep) → interleave, count cycles, flag clock changes | `runs/kernel/2026-11-22_12-02-39-registers-sweep` |
| 2026-11-22 | **Why the clock falls: power, not spills.** The baseline has no spills; across the variants the clock follows the power draw. `nvidia-smi`'s clock reasons: FP64 and 1-SM runs stop at the voltage ceiling (2880–2895 MHz, `reliability`); FP32 on 36 SMs hits `sw_power_cap`, under an enforced limit that moved 85–115 W within one run. Locking the clock (`nvidia-smi -lgc`) is a driver setting, left to the user | [walkthrough.md piece 21](walkthrough.md#piece-21-the-code-mapped-to-the-hardware-measured-instruction_mixpy-kernel_experimentspy-gpu_monitorpy) |
| 2026-11-22 | **The open questions reviewed** (section 0, Sunday): Q1 the FP32 kernel's FP64 instructions are `sinf`/`cosf`'s slow path, never run; Q2 the `NOP`s are waits for the FP64 unit, not padding (guess wrong); Q3 the 5× slower pass was noise on a 20 ms kernel, not the idle clock (guess wrong); Q4 the CPU's slower sweep is not explained, and the shared-power guess is not supported: the one-core baseline varies ~25% between sessions; I1 stray observer files (committed by accident) removed | [walkthrough.md pieces 20, 21, 23](walkthrough.md#piece-21-the-code-mapped-to-the-hardware-measured-instruction_mixpy-kernel_experimentspy-gpu_monitorpy) |
| 2026-11-21 | **Steps 7–9 committed** (`a15ef84`): code, docs, 198 sweep points, the scaling figures and report, the showcase's stills and SNR (videos and raw images kept out of git). **A slip, repaired:** a test script's `rm -rf runs/*kernel-experiments` had deleted step 6's committed report; restored from `d8ccbb5` in `4b518fb` | — |
| 2026-11-21 | **Step 9 done: the showcase.** 1024², 300 frames, 4 observers in one pass (5 GB on the GPU), N = 3.9·10¹² photons in 385 s (1.0·10¹⁰ photons/s, FP32; ~4 days on one CPU core). SNR measured = predicted: center 14.1, limb ring 11.7. Four 10 s mp4s (kept out of git). **Amdahl's ceiling:** strong FP32 at N = 10⁸, whole run p = 0.58, ceiling 2.4, 36 SMs give 2.3× | [walkthrough.md piece 24](walkthrough.md#piece-24-the-showcase-showcasepy) |
| 2026-11-21 | **The sweeps' results.** Strong, FP64: 35.9× on 36 SMs (photon phase), p = 0.9998; whole run p = 0.9978, s fitted 0.0022 vs measured 0.0019; the clock steady at 2880 MHz. Strong, FP32: 33.1× in seconds but 36.0× in cycles: the clock falls to 2636 MHz on 36 SMs. Weak, FP64: flat, Gustafson s′ fitted 0.054 vs measured 0.058; FP32: efficiency 0.92 on 36 SMs (the clock). **Correction:** the first weak scaled speedup counted the serial part n times (negative s′); now Gustafson's definition. Added: a small-N strong sweep (N = 10⁸) for Amdahl's ceiling | [walkthrough.md piece 23](walkthrough.md#piece-23-the-scaling-analysis-scalingpy) |
| 2026-11-21 | **Step 9 prepared: `showcase.py`** (N from the SNR wanted at the limb, the run on the GPU, mp4 per observer read frame by frame with `np.memmap`, measured vs predicted SNR; tested on synthetic frames). **`ffmpeg` 9.0.2 installed** in the `aga511` env from conda-forge only (`--override-channels`; Anaconda's default channels asked for their Terms of Service, not accepted). Proposed run: 1k, 300 frames, SNR 10, FP32, N = 3.9·10¹², ~7 min, after the sweeps | — |
| 2026-11-21 | **Step 8: `scaling.py`** (Amdahl and Gustafson fits, p(N), devices; figures with their data). First results: **p(N) in FP32 grows from 0.14 (10⁵) to 0.9993 (10⁹)** for the photon phase, and from ≈ 0 to 0.987 (10¹⁰) for the whole run; GPU vs one CPU core at the same precision, N = 10⁸: 82× (FP64), 850× (FP32) on the photons, ~28× on the whole run | [walkthrough.md piece 23](walkthrough.md#piece-23-the-scaling-analysis-scalingpy) |
| 2026-11-21 | **Step 7: `sweeps.py` built and running** (one engine process per point, calibrated N, interleaved repeats, warm-up, resume; tested). `engine.default_star_config` shared with `kernel_experiments.py`. The full set (devices, problem-size, weak and strong in FP64 and FP32) started: ~50 min. Step 6 committed `d8ccbb5` | [walkthrough.md piece 22](walkthrough.md#piece-22-the-sweeps-sweepspy) |
| 2026-11-21 | **Step 6 done: the code mapped to the hardware, measured** (build variants, laps in `trace_photon`, CUDA-event kernel time, `gpu_monitor.py`, `instruction_mix.py`, `kernel_experiments.py`). The kernel is limited by arithmetic: direction 33–41%, surface 20–28%, `atan2` 22% in FP64, the image's atomics 13–15%. The Scene in global memory: same speed (suspect cleared). Capping registers raises occupancy to 1.0 but is up to 9% slower (spills). FMA: +7% in FP64. FP64 is 17× slower than FP32 per photon on a 1:64 GPU, because integer work is the largest class. FP32 runs at a lower clock (2310 vs 2812 MHz). Step 5 committed `fbeff8c` | [walkthrough.md piece 21](walkthrough.md#piece-21-the-code-mapped-to-the-hardware-measured-instruction_mixpy-kernel_experimentspy-gpu_monitorpy) |
| 2026-11-21 | **Step 5 done: memory, passes and the kernel's resources** in every output (`ProcessorHardware`, `KernelResources`). **Registers limit the occupancy:** 80 registers (FP64) → 768 threads per SM = 0.50; 67 (FP32) → 0.58. Passes chosen from the GPU's free memory when not given (a 10 GB job → 4 + 4). Copy back ~24 GB/s; at 1k the save (disk, `fsync`) dominates the run. Steps 3–4 committed `3a230c7` | [walkthrough.md piece 20](walkthrough.md#piece-20-memory-passes-and-the-kernels-resources-gpu_devicecu-gpucu-optionscu-reportcu) |
| 2026-11-21 | **Step 4 done: the SM mask** (`gpu_sms.cu`, EP3's `%smid` design, cooperative launch, `--sms N`) and **work per SM** (`SM` and `SMS` lines: chunks, photons, recorded, cycles). The same image on any number of SMs; the queue balances the SMs within 0.1%. **Finding: the GPU's clock falls from 2.88 GHz on 1 SM to 1.95 GHz on 36**, so 36 SMs are 30.7× one SM in cycles but 20.8× in seconds. One SM ≈ 1.25 CPU cores in FP64, ~25 in FP32 | [walkthrough.md piece 19](walkthrough.md#piece-19-the-sm-mask-and-the-work-per-sm-gpu_smscu) |
| 2026-11-21 | **Step 3 done: the star runs on the GPU.** `gpu.cu` (constant Scene, EP3's queue around the shared `trace_photon`, occupancy launch, pinned host buffer), `gpu_image.cu` (atomics), `gpu_rng_test.cu` (split from `gpu.cu`), `star.py --compare-gpu`. **GPU = CPU bit for bit in FP64**; FP32: 54 pixels (3.2·10⁻⁵), 12 without FMA. 78 / 67 registers, no spills. FP64 6.3·10⁸ photons/s (75× one core), FP32 9.3·10⁹ (14.6× FP64). At N = 10⁸ FP32 the run is 90% serial (context creation) | [walkthrough.md piece 18](walkthrough.md#piece-18-the-star-on-the-gpu-gpucu-gpu_imagecu) |
| 2026-11-21 | **Review before step 3** (all 25 files against the plan and the style): `photon.cuh` no longer includes the CPU generator; the Random123 test moved to `rng_test.cu`, so `main.cu` is only the pass loop; the build check cached in `engine.py` (100 checks in 3 ms); the docs use the code's names. Checks unchanged. Committed `58e40b3` with step 2 | [gpu_engine.md](gpu_engine.md) |
| 2026-11-21 | **Step 2 done:** cuRAND on the GPU = our CPU copy, bit for bit (raw and uniform, FP64 and FP32, up to chunk 2³² + 5). Found: our Philox constants redefined cuRAND's macros, renamed `PHILOX_KEY_WEYL_*`, `PHILOX_ROUND_MULTIPLIER_*` | [walkthrough.md piece 17](walkthrough.md#piece-17-random-numbers-on-the-gpu-gpu_rngcuh-gpucu) |
| 2026-11-21 | **Step 1 done:** `processor.cuh` with `cpu.cu` (the loop, moved) and a first `gpu.cu` + `gpu_device.cu`; the Makefile removed, `engine.py` builds the four binaries (4 s each for the CPU; cached when the code hash matches); `COPY_TIME` and the processor's description in the output; all CPU checks identical to before (split, passes bit for bit; χ²/dof 0.9957; FP32 1.2·10⁻⁴) | [walkthrough.md piece 16](walkthrough.md#piece-16-the-processor-split-the-pass-loop-stops-knowing-which-processor-traces) |
| 2026-11-21 | **No Nsight needed:** IAG may not have it and the laptop's counters are blocked. EP4 measures itself: CUDA events, attributes, occupancy, per-SM counters, `cuobjdump`, `nvidia-smi`, and section timers with `clock64()` in a separate build (so the measuring does not change the timed binaries) | [hpc_architecture.md §8](hpc_architecture.md#no-nsight-needed-decided-2026-11-21) |
| 2026-11-21 | **The code mapped to the hardware, and observability:** each piece of the photon → its unit and memory space; measured by the compiler's report, the instruction mix (`cuobjdump`), occupancy, per-SM counters, `cudaMemGetInfo`, `nvidia-smi` sampling, `nsys`. Suspect found: `slot_of_imu` in constant memory is read with a different index per thread. CPU cores deferred; the CPU keeps the same checks | [hpc_architecture.md §7–8](hpc_architecture.md#7-the-code-mapped-to-where-it-runs-the-gpu-one-photon) |
| 2026-11-21 | **The hardware is the architecture (option B):** `processor.cuh`, `cpu.cu`, `gpu.cu`, `cpu_cores.cu`, `gpu_sms.cu`; levels node → processor → compute units → lanes → memory spaces, each a scaling axis; the same work queue on CPU and GPU; benchmarking after Hoefler & Belli (state every baseline) | [hpc_architecture.md](hpc_architecture.md) |
| 2026-11-21 | **No Makefile:** `engine.py` builds with one `nvcc` call per binary; the code hash decides when to rebuild; binaries named `star_cpu64`, `star_cpu32`, `star_gpu64`, `star_gpu32` | [gpu_engine.md §3](gpu_engine.md#3-building-from-python-no-makefile-decided-2026-11-21) |
| 2026-11-21 | **All of EP4 today (Saturday); the class prep on Sunday.** Concepts used while building go to `cuda_class/ep4_concepts.md` | [section 0](#0-today-all-of-ep4-saturday-2026-11-21-to-review) |
| 2026-11-21 | **Before-GPU changes done:** first touch in SETUP, `fsync`, balanced passes, binary units, `HOST=`, no spaces, `--describe`; all checks pass | [backlog C′](#c-before-the-gpu-engine-changes-the-gpu-will-inherit) |
| 2026-11-21 | **Finding:** `malloc` + `memset(0)` did not touch the pages (probably folded into `calloc` by the compiler); a `volatile` write per page does | [backlog C′](#c-before-the-gpu-engine-changes-the-gpu-will-inherit) |
| 2026-11-21 | **Correction:** `fsync` costs 0.31–0.43 s per 315 MB in the engine (new files), not 0.15–0.18 s (the `dd` test overwrote one file) | [walkthrough.md piece 13](walkthrough.md#questions-on-piece-13) |
| 2026-11-21 | **Builds: SASS only, no PTX, built on the machine that runs it,** for its exact compute capability. A wrong build fails loudly; no JIT time in the serial part; the toolkit matches the driver | [gpus.md](gpus.md#what-the-code-must-do-to-scale-there-unchanged) |
| 2026-11-21 | **Finding (to confirm):** FP32 moves 1.2·10⁻⁴ of the photons now vs 5·10⁻⁴ before, probably because the pole-on observer is farther from the pole | [jobs_and_passes.md §15](jobs_and_passes.md#what-we-learned) |

---

## 4. Backlog: tasks to do

Collected from the design docs and the walkthrough. New tasks found on the way are added here, marked with where they came from.

### C′: before the GPU (engine changes the GPU will inherit)

Found in the walkthrough of phase C ([recap](walkthrough.md#recap-of-pieces-1015-what-we-learned)). All small; done once on the CPU, the GPU engine gets them for free. **All done 2026-11-21:**

- [x] **Commit phase C** as it was: `5db4d99`.
- [x] *(piece 13)* **Touch the pages at allocation** (`touch_every_page`, inside SETUP). Every pass now zeroes in the same time. 512², 300 frames, 4 passes: SETUP 0.13–0.15 s (the first touch), `ZERO_TIME` 0.011–0.016 s in every pass (pass 1 was 0.15–0.21 s).
  - **Found on the way:** zeroing with `memset` right after `malloc` did *not* touch the pages (SETUP stayed at 0.0005 s). Most likely the compiler turned `malloc` + `memset(0)` into `calloc`, which for a big block takes fresh pages that are already zero and writes nothing. Writing one count per page through a `volatile` pointer cannot be folded away.
- [x] *(piece 13)* **`fsync` before the rename**, and of the folder after it (`close_on_disk`, `sync_parent_folder` in `files.cuh`).
  - **Measured in the engine: 0.31–0.43 s per 315 MB** (~0.8–1 GB/s), not the 0.15–0.18 s of the `dd` test: `dd` overwrote the same file, the engine writes new files. Small in a run of hours; in short checks SAVE is now a visible part of the serial time.
- [x] *(piece 12)* **Balanced passes:** K′ = ⌈observers / passes⌉. `--observers-per-pass 3` with 4 observers now gives 2 + 2 (48 MB instead of 72).
- [x] *(piece 12)* **Binary units:** `KiB`, `MiB`, `GiB`, `TiB` (`70MiB` → K = 2, as `0.07GB`).
- [x] *(piece 14)* **`HOST=`** in the summary, the `CONFIG` line and the records.
- [x] *(piece 14)* **Spaces rejected** in `--output-dir`, `--fingerprint`, `--code-hash`.
- [x] *(piece 15)* **`--describe`:** a `CONFIG` line (the derived configuration, `OBSERVER_BYTES`, `IMAGE_BYTES`) and one `OBSERVER` line per observer (μ bin, bin center, its pass, its file name), then exit. `star.py` takes EP2's `nmu` from it; `ep2_nmu` is gone.
- [x] **All checks rerun:** split and passes bit for bit (also 5 observers in passes of 3 → 3 + 2); EP4 vs EP2 χ²/dof = 0.9957; FP32 vs FP64 1.2·10⁻⁴ moved, 1.51× faster.
- [ ] *(new, decide in G)* **Sweeps and the disk:** with `fsync`, SAVE is real disk time, part of the serial fraction. Should scaling sweeps include it (the real program), or record without saving (`--no-save`) to measure the GPU alone? Probably both, as two experiments.

### C: follow-ups that can wait
- [ ] *(piece 15)* **Analysis one observer at a time** (`np.memmap`, frame by frame). Not needed until images reach ~1 GB per observer, so before phase H's big runs, not before the GPU.
- [ ] A dedicated FP32 run for pole-on observers, to confirm the 1.2·10⁻⁴ vs 5·10⁻⁴ finding. Before the monograph.

### Preflight: can we run here? (local and remote) *(asked 2026-11-21)*

Before any run or sweep, on the laptop or the A30, check that the machine can do it, so a session at IAG is not lost to a missing tool or a busy GPU. One command, `check system` locally and `remote check system <host>` on the server ([driver.md §2](driver.md#2-the-commands)), and run automatically by `run` and `sweep` before they launch.

| # | Check | How | Fails when |
|---|---|---|---|
| 1 | **The GPU is visible** | `nvidia-smi`: name, driver version, the CUDA version the driver supports | no GPU, no driver (in WSL: `/usr/lib/wsl/lib` missing from the path) |
| 2 | **The toolkit fits the driver** | `nvcc --version` vs the driver's CUDA version | the toolkit is newer than the driver supports |
| 3 | **The engine runs on this GPU** | the GPU engine's `--test-device`: device count, compute capability, a tiny kernel | built for another architecture ("no kernel image"), e.g. `-arch=native` from the laptop on the A30 |
| 4 | **Same numbers as the CPU** | `--test-device` also checks cuRAND's first numbers against `rng.cuh`, and a tiny star bit for bit against the CPU engine | any difference: the GPU result cannot be trusted |
| 5 | **Enough GPU memory, now** | `cudaMemGetInfo`: free vs the job's budget + headroom; MIG mode and slice size | another job holds the memory; a MIG slice is smaller than expected |
| 6 | **Nobody else is using it** | `nvidia-smi --query-compute-apps`, utilization | another process on a shared A30: our timings (and Amdahl's p) would be wrong. A warning, not a stop |
| 7 | **Room on the host** | RAM for one observer's buffer; free disk ≥ the job's output (from `plan`); the job folder is writable | a full disk halfway through a run of hours |
| 8 | **Remote only** | SSH reaches the host (inside the IAG network); the repo path; `nvcc`, `rsync`, `setsid`; time limits or a scheduler | not on the IAG network; a tool missing |
| 9 | **Profiling allowed** (optional) | `ncu` on a tiny kernel | `ERR_NVGPUCTRPERM`: performance counters not allowed (as on the laptop today) |

- **Output:** one line per check (`CHECK NAME=… RESULT=PASS|WARN|FAIL DETAIL=…`), saved with the machine's profile in `machines/`, so the A30's last check can be read from the laptop.
- **`run` refuses on a FAIL** and shows the WARNs; `--skip-checks` for when you know better.
- **Builds on** EP3's `buffon -i` (device query) and the backlog's `remote setup` and `machine probe`, which it partly replaces.

Tasks:
- [ ] Engine (phase F): `--test-device` (checks 3–5), printing `CHECK` lines.
- [ ] Driver (phase D): `check system` running checks 1–7 locally; `run` and `sweep` call it first.
- [ ] Remote (phase E): `remote check system <host>`: the same checks over SSH, plus check 8.
- [ ] Try it on the laptop now (checks 1, 2, 7 need no GPU engine), and at IAG on the first visit.

### D: the local driver (the part F and G need first)
- [ ] Split `star.py` into the `driver/` package ([driver.md §8](driver.md#8-modules)), starting with `engine.py` (command, output lines, records).
- [ ] Job folder, `job.json`, fingerprint, code hash; job state from the records (new / partial / complete / interrupted).
- [ ] `run` (continue by default, `--rerun` with the backup question, `--fresh`), `status`, `clean`.
- [ ] Presets (512, 1k, 2k, 4k) and job files in TOML; prompts.
- [ ] `plan`: footprint, K, passes, N ↔ SNR; time from a calibration.
- [ ] `--detach` with `run.pid`, `stop`.

### E: remote (the part G needs first)
- [ ] `hosts.py` (`LocalHost`, `SSHHost`), tested against `localhost`.
- [ ] `remote setup` (sync, build with `ARCH`), `remote run`, `remote status`, `remote fetch`.
- [ ] `machine probe / calibrate`, local and remote.

### F: the GPU engine
- [ ] The build in `engine.py` (no Makefile, [gpu_engine.md §3](gpu_engine.md#3-building-from-python-no-makefile-decided-2026-11-21)): the CUDA build next to the CPU ones. **SASS only, no PTX, built on the machine that runs it** ([decided](gpus.md#what-the-code-must-do-to-scale-there-unchanged)): `-arch=native` by default, `--arch 80` for the A30, as `-gencode arch=compute_80,code=sm_80`.
- [ ] `BUILD_CC`, `CUDA_TOOLKIT`, `CUDA_DRIVER` (versions) in the summary and the records.
- [ ] *(decide)* **Bit for bit across GPUs:** the same source compiled for `sm_80` and `sm_120` may fuse multiply-adds differently (FMA), so a few photons could land in other pixels. `--fmad=false` makes the arithmetic identical at some cost in speed. Measure both: the laptop vs the A30 image, with and without it.
- [ ] The Scene in `__constant__` memory (`cudaMemcpyToSymbol`).
- [ ] `DeviceImage` + `count_photon_atomic` (`atomicAdd`), zeroed with `cudaMemset`, copied back per observer.
- [ ] cuRAND Philox on the device, one subsequence per chunk; check the first numbers against `rng.cuh`, then the whole image bit for bit against the CPU.
- [ ] The kernel: persistent threads taking chunks from a device work queue (`atomicAdd` on a counter), launch size from the occupancy API (as EP3's `choose_launch`).
- [ ] Passes on the device; the memory budget from `cudaMemGetInfo`; `GPU_FREE_BYTES` in the output.
- [ ] The SM mask (`%smid`, as EP3) for the Amdahl sweep.
- [ ] Progress lines from the GPU run *(open question: how the host learns how many chunks are done while the kernel runs)*.
- [ ] *(recap)* Per pass: `cudaMemcpyToSymbol` of the 16 KB `slot_of_imu` table; `cudaMemset` of the slots in use; one `cudaMemcpy` per observer into a host buffer of **one** observer (not K), reused.
- [x] `PROCESSOR`, `PROCESSOR_NAME` (the GPU's name), `COMPUTE_CAPABILITY`, `COMPUTE_UNITS`, `UNIT` in the lines and records *(step 1)*.

### G: sweeps and scaling
- [ ] Sweep files and `sweeps.py` (points as records, repeats).
- [ ] `scaling.py`: strong with the Amdahl fit, p(N), weak with the Gustafson fit, side by side, the serial part measured vs fitted, sizes, machines.

### H: the showcase
- [ ] `video` (mp4 through ffmpeg), `pictures` with the measured SNR.
- [ ] The rotation length (sets the frames at 30 fps).
- [ ] The big runs at IAG.

### I: the deliverables
- [ ] Monograph outline: the technique, the implementation, the performance tests.
- [ ] The 10–15 min presentation.

---

## 5. Parked, not forgotten

- [ ] Nsight Compute on Buffon: enable the GPU performance counters on Windows, restart, fill [nsight.md §4.2](../cuda_class/nsight.md).
- [ ] Walkthrough piece 16 (the docs and the class plan).
- [ ] The class: date, length, the team's split ([class_plan.md](../cuda_class/class_plan.md#open-questions)).
- [ ] The questions for IAG ([gpus.md](gpus.md#questions-for-iag)).
- [ ] Commit: everything since `d945e97` (2026-11-21) is uncommitted.

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
| 5 | **F5: memory, passes and the kernel's resources** | `cudaMemGetInfo` (before, after the context, after the image) → `--memory-budget` default; passes (the loop stays on the host); `cudaFuncGetAttributes` (registers, local memory = spills, constant bytes), occupancy, the GPU's attributes (L2, bandwidth, FP32:FP64); build and driver versions | the passes check on the GPU: bit for bit; the resource keys in the output | device memory, registers and occupancy, host ↔ device traffic |
| 5b | ~~The CPU's cores~~ **deferred** | the CPU stays one thread; it runs all the checks and is the one-core baseline, said so ([hpc_architecture.md §8](hpc_architecture.md#8-observability-what-we-measure-and-how)) | — | — |
| 6 | **The map and the speed** | **the code mapped to the hardware** ([hpc_architecture.md §7](hpc_architecture.md#7-the-code-mapped-to-where-it-runs-the-gpu-one-photon)): the compiler's report and `instruction_mix.py` (FP64 vs FP32: integer, FP64, FP32, special functions, atomics, constant, spills); `nvidia-smi` sampled during runs (clocks, power, temperature, memory); CPU vs GPU, FP64 vs FP32 at a fixed N; **the `slot_of_imu` experiment** (constant vs global memory); CUDA events for the GPU-side times; **our own section timers** (`star_gpu64_sections`, cycles per piece of the photon). No Nsight needed | the map confirmed or corrected; the numbers; cycles per section, FP64 vs FP32 | where each line runs, FP64 units vs FP32 cores vs SFU vs L2 atomics, constant-cache broadcast vs serialization, divergence, clocks, measuring without a profiler, the observer effect |
| 7 | **G1: the sweeps** | `sweeps.py`, defined in [gpu_engine.md §7](gpu_engine.md#7-the-sweeps-today-to-review): `strong` and `weak` over SMs 1–36 (FP64 and FP32), `problem-size`, `devices` (the GPU against one CPU core, stated); N calibrated; 3 repeats; ~1 h in the background | the data on disk | strong vs weak scaling, p(N) |
| 8 | **G2: the scaling analysis** | Amdahl fit (p), Gustafson fit (s′), efficiency, strong and weak side by side, the serial part measured (SETUP + ZERO + SAVE) vs fitted | the plots | Amdahl, Gustafson, serial fraction |
| 9 | **H: the showcase** | one big run (size, frames and N set from the measured speed and the SNR formula), gifs (and mp4 if `ffmpeg` is there); pictures made one observer at a time | the star's video | SNR ↔ N, photons per pixel, μ bins |
| 10 | **Wrap up** | EP4 README results, the roadmap's log, commit | committed | — |

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

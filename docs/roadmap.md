# EP4 roadmap: where we are and how we got here

**The one place for the plan.** Phases are lettered (A, B, C…) so they don't get mixed up with the walkthrough's pieces (1–10) or the old step lists in [jobs_and_passes.md §9](jobs_and_passes.md#9-implementation-order) and [driver.md §9](driver.md#9-implementation-order), which now point here.

**The goal (TCD, due 2026-12-01):** EP2's spotted star rebuilt with CUDA; a monograph (the technique, the implementation, performance tests: Amdahl, speedup, Gustafson); a 10–15 min presentation. Separately, the CUDA class for AGA 511.

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

**The order to decide.** As designed, D and E come before F. But F is the heart of the TCD, and G (the Amdahl and Gustafson results) needs F. With the deadline on 2026-12-01, a possible order is **F → G → I**, with D and E reduced to what F and G need (running sweeps and fetching results). D and E in full can come after.

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
| 2026-11-21 | **Builds: SASS only, no PTX, built on the machine that runs it,** for its exact compute capability. A wrong build fails loudly; no JIT time in the serial part; the toolkit matches the driver | [gpus.md](gpus.md#what-the-code-must-do-to-scale-there-unchanged) |
| 2026-11-21 | **Finding (to confirm):** FP32 moves 1.2·10⁻⁴ of the photons now vs 5·10⁻⁴ before, probably because the pole-on observer is farther from the pole | [jobs_and_passes.md §15](jobs_and_passes.md#what-we-learned) |

---

## 4. Backlog: tasks to do

Collected from the design docs and the walkthrough. New tasks found on the way are added here, marked with where they came from.

### C′: before the GPU (engine changes the GPU will inherit)

Found in the walkthrough of phase C ([recap](walkthrough.md#recap-of-pieces-1015-what-we-learned)). All small; done once on the CPU, the GPU engine gets them for free. In the proposed order:

- [ ] **Commit phase C** as it is: a safe point before changing it.
- [ ] *(piece 13)* **Touch the pages at allocation:** zero the image once in `allocate_image` (inside SETUP), so every pass's `ZERO_TIME` is pure zeroing. On the GPU, `cudaMalloc` + `cudaMemset` behave the same way, so the two engines' phase times mean the same thing. Today pass 1's `ZERO_TIME` is 25–30× the others.
- [ ] *(piece 13)* **`fsync` before the rename,** so a finished file survives a power cut, not only a crash. `SAVE_TIME` then becomes the real disk time (today: time to the OS's cache, ~2 GB/s). *Proposed: yes. Measured: 315 MB in 0.15–0.18 s with `fsync` vs 0.08–0.16 s without ([what fsync is](walkthrough.md#questions-on-piece-13)).*
- [ ] *(piece 12)* **Balanced passes:** after the number of passes is known, use K′ = ⌈observers / passes⌉ (3 + 1 becomes 2 + 2: same time, less memory).
- [ ] *(piece 12)* **Binary units** in `--memory-budget`: `GiB`, `MiB` (what `nvidia-smi` shows).
- [ ] *(piece 14)* **`HOST=`** (from `gethostname`) in the summary and the records.
- [ ] *(piece 14)* **Reject spaces** in `--output-dir`, `--fingerprint`, `--code-hash`, which would break `KEY=VALUE`.
- [ ] *(piece 15)* **`--describe`:** the engine prints a `CONFIG` line (nmu, each observer's μ bin and bin center, K, passes, footprint) and exits without simulating. The driver reads the derived values instead of recomputing them (fixes "nmu derived twice"), and `plan` gets the exact numbers from the engine itself.
- [ ] Rerun all the checks (`--check-split`, `--check-passes`, `--compare-ep2`, `--compare-precision`) after these changes.

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
| 8 | **Remote only** | SSH reaches the host (inside the IAG network); the repo path; `make`, `rsync`, `setsid`; time limits or a scheduler | not on the IAG network; a tool missing |
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
- [ ] `Makefile`: a CUDA build next to the CPU ones. **SASS only, no PTX, built on the machine that runs it** ([decided](gpus.md#what-the-code-must-do-to-scale-there-unchanged)): `-arch=native` by default, `ARCH=sm_80` for the A30, as `-gencode arch=compute_XX,code=sm_XX`.
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
- [ ] `DEVICE` = the GPU's name in the lines and records.

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

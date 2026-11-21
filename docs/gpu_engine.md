# The GPU engine (design)

What we build today, as files and components. **Status (2026-11-21):** step 1 built (`processor.cuh`, `cpu.cu`, `gpu.cu` without photons yet, `gpu_device.cu`, `engine.py`); the rest in progress. `star.py` takes `--processor cpu|gpu`.

The rule from the code style: **the file names and function names are the documentation.** Each file holds one thing, named for what it holds; the reasoning lives here.

## 1. The idea in one picture

The pass loop (`main.cu`) does not change. It stops knowing *who* traces the photons:

```
main.cu: options → scene → observers → for each pass:
                                          load the pass        ┐
                                          zero the image       │  a processor:
                                          trace the photons    │  cpu.cu  (today's loop)
                                          hand each observer   │  gpu.cu  (the kernel)
                                          to the host          ┘
                                        save, record, report  (shared)
```

- **`processor.cuh`** is the list of what the pass loop asks of a CPU or a GPU.
- **Two implementations,** chosen at link time by the build (`engine.py`, section 3): `star_cpu64` links `cpu.cu`, `star_gpu64` links `gpu.cu`. No `#ifdef` in the pass loop.
- **The physics is shared:** `photon.cuh`'s `trace_photon` already takes the random generator as a template, so the GPU calls the same function with cuRAND's state.
- **Recording decision B** (2026-11-21) is kept: the CPU records with `Image` + `count_photon`, the GPU with its own `GpuImage` + `count_photon_atomic`.

## 2. The files

### Engine (C++/CUDA, `EP4/engine/`)

| File | New / changed | What it holds |
|---|---|---|
| `processor.cuh` | **new** | the interface of a processor: `open_processor`, `load_pass`, `zero_pass_image`, `trace_pass`, `observer_on_host`, `processor_image_bytes`, `describe_processor` (fills a `ProcessorInfo`), `compare_rng_with_cpu`, `close_processor`. `Processor` is opaque (each implementation defines it) |
| `cpu.cu` | **new** (moved from `main.cu`) | the CPU as a processor: `Image`, `PhiloxState`, `simulate_chunk`, `simulate_chunks`, progress. Today's loop, moved, not rewritten |
| `cpu_cores.cu` | **new** (step 5b) | the CPU's cores: one thread per allowed core, taking chunks from the same queue as the GPU ([hpc_architecture.md §3](hpc_architecture.md#3-one-design-at-every-level-the-work-queue)) |
| `gpu.cu` | **new** | the GPU as a processor: the Scene in `__constant__` memory, the kernel `trace_photons`, the queue (`take_next_chunk`, `take_chunks_until_none_left`), `choose_launch`, `run_kernel`, `GpuCounters` |
| `gpu_image.cuh`, `gpu_image.cu` | **new** | `GpuImage` in device memory: `allocate_gpu_image`, `zero_gpu_image` (`cudaMemset`), `copy_observer_to_host` (`cudaMemcpy` into a host buffer of one observer), `free_gpu_image`; `count_photon_atomic` (`atomicAdd`) |
| `gpu_rng.cuh` | **new** | cuRAND's Philox as `trace_photon`'s generator: `uniform(curandStatePhilox4_32_10_t *)`, the GPU twin of `rng.cuh`'s `uniform(PhiloxState *)` |
| `gpu_sms.cuh`, `gpu_sms.cu` | **new** | choosing the SMs, as EP3: `sm_id` (`%smid`), `list_sm_ids`, `allow_first_sms`, `resolve_sm_count`, `is_allowed` |
| `gpu_device.cuh`, `gpu_device.cu` | **new** | the GPU itself: `check` (CUDA errors), properties, name (spaces → `_`), compute capability, free memory, runtime and driver versions → a `ProcessorInfo` |
| `main.cu` | changed | only the pass loop, calling the processor |
| `rng_test.cuh`, `rng_test.cu` | **new** (moved from `main.cu`) | `--test-rng`: the Random123 known answers, chunk 0's first numbers, and the processor's `compare_rng_with_cpu` |
| `run.cuh` | changed | `ProcessorInfo` (kind, name, compute capability, compute units, unit; more in steps 4–5) replaces the `DEVICE_NAME` constant; `PassTimes` gains `copy` |
| `report.cu` | changed | prints the `ProcessorInfo` keys and `COPY_TIME` |
| `options.cu` | changed | `--sms N` (GPU) and `--cores N` (CPU): the compute units allowed; each processor refuses the other's |
| `image.cuh`, `image.cu` | changed | saving takes a block of counts (`save_counts`, `photons_in`), so the GPU's host buffer and the CPU's image are saved by the same code |
| `Makefile` | **removed** | the build moves to `engine.py` (section 3) |
| `real.cuh`, `rng.cuh`, `geometry.cuh`, `scene.*`, `observers.*`, `photon.cuh`, `files.cuh`, `report.cuh` | unchanged | |

### Driver and analysis (Python, `EP4/`)

| File | New / changed | What it holds |
|---|---|---|
| `engine.py` | **new** (moved from `star.py`) | **building the engine** (section 3) and talking to it: which binary (`cpu`/`gpu` × 64/32), command, run, `--describe`, records, loading observers **one at a time** |
| `star.py` | changed | the command line: a run with pictures, and the checks; `--processor cpu\|gpu`, `--sms`; new check **`--compare-gpu`** (CPU vs GPU on the same photons) |
| `analysis.py` | changed | pictures one observer at a time (a 1k observer is 1.3 GB) |
| `sweeps.py` | **new** | running the scaling sweeps: `strong` (fixed N) and `weak` (N ∝ SMs) over a list of SM counts, with repeats; one result file per point, existing points skipped |
| `scaling.py` | **new** | the analysis of the sweeps: the Amdahl fit (p), the Gustafson fit (s′), efficiency, strong and weak side by side, the serial part measured vs fitted; figures with their data |

This is the first slice of the driver of [driver.md §8](driver.md#8-modules): `engine.py`, `sweeps.py`, `scaling.py` are its modules, flat in `EP4/` for now. The `driver/` package comes with the full phase D, later.

## 3. Building, from Python (no Makefile) [decided 2026-11-21]

`engine.py` builds the four binaries itself, with one `nvcc` call each:

| Binary | Sources | Flags |
|---|---|---|
| `star_cpu64` | shared + `cpu.cu` | |
| `star_cpu32` | shared + `cpu.cu` | `-DSTAR_FP32` |
| `star_gpu64` | shared + `gpu.cu gpu_image.cu gpu_sms.cu gpu_device.cu` | the GPU architecture |
| `star_gpu32` | the same | the GPU architecture, `-DSTAR_FP32` |
| `star_gpu64_sections`, `star_gpu32_sections` | the same | `-DSTAR_SECTION_TIMERS`: cycles per piece of the photon ([hpc_architecture.md §8](hpc_architecture.md#no-nsight-needed-decided-2026-11-21)); never used for timing the sweeps |

```
shared  = main.cu options.cu scene.cu observers.cu image.cu report.cu rng_test.cu
command = nvcc -O2 -std=c++17 -Xcompiler -Wall <architecture> [-DSTAR_FP32] <sources> -o build/<binary>
```

- **The architecture:** `-arch=native` by default (the GPU of the machine that builds); `--arch 80` gives `-gencode arch=compute_80,code=sm_80` for the A30. SASS only, no PTX ([gpus.md](gpus.md#what-the-code-must-do-to-scale-there-unchanged)).
- **Rebuild only when something changed:** the **build hash** = a hash of the engine's sources (`*.cu`, `*.cuh`) + the `nvcc` command + `nvcc --version`. It is stored next to the binary (`build/star_gpu64.hash`); a different hash means rebuild. This is the **code hash** the records carry ([jobs_and_passes.md §5](jobs_and_passes.md#code-changes-decided)): one hash, two uses.
- **Binaries in `engine/build/`**, ignored by git.
- **The names say what they are:** device and precision (`star_gpu32`), replacing `star` and `star32`.
- **By hand:** `python engine.py` builds all four and prints each `nvcc` command, which can also be copied and run directly. Handy to show the build in class.
- **Remote:** the driver sends the same `nvcc` command over SSH. The server needs neither Python nor `make`.

## 4. A pass on the GPU

```
load_pass        cudaMemcpyToSymbol(gpu_scene, &scene)       the Scene (~18 KB) into constant memory
zero_pass_image  cudaMemset(the slots in use)                ZERO_TIME
trace_pass       reset the counters; launch trace_photons;   PHOTONS_TIME
                 cudaDeviceSynchronize; read the counters
for each observer:
  observer_on_host   cudaMemcpy one block → host buffer      COPY_TIME   (new phase)
  save_counts        the .partial, fsync, rename             SAVE_TIME   (shared with the CPU)
  record             the .record                             (shared)
```

**The kernel, as EP3's `throw_needles`:**

```
trace_photons (one per thread, persistent):
    if this SM is not allowed: return                        the SM mask (Amdahl)
    chunk = take_next_chunk()                                the work queue: atomicAdd on a counter
    while chunk < chunks:
        curand_init(seed, chunk, 0, &rng)                    chunk = subsequence, as on the CPU
        for each photon of the chunk:
            pixel = trace_photon(&gpu_scene, &rng)           the shared physics
            tally it; if recorded: count_photon_atomic       atomicAdd on the image
        chunk = take_next_chunk()
    add this thread's tallies to the counters                one atomic per counter per thread
```

- **The constant Scene and the kernel live in the same file** (`gpu.cu`): without separate compilation (`-rdc`), a `__constant__` variable is visible only in its own file.
- **cuRAND's device API is header-only** (`curand_kernel.h`): nothing to link.
- **Photon counts:** each thread tallies in registers and adds them once at the end, so the counters cost ~4 atomics per thread, not per photon.
- **The host buffer holds one observer,** reused for each: host memory stays one footprint whatever K is.

## 5. The output, new keys

- `PROCESSOR` = `CPU` or `GPU`; `PROCESSOR_NAME` = its name with `_` for spaces (`NVIDIA_GeForce_RTX_5070_Laptop_GPU`); `COMPUTE_UNITS` and `UNIT` (`core` or `sm`). *(built in step 1)*
- `COMPUTE_CAPABILITY`, `SMS` (allowed), `SMS_USED` (seen by the kernel), `SM_COUNT`, `BLOCKS`, `THREADS_PER_BLOCK`, `REGISTERS_PER_THREAD`, `CYCLES` (the slowest block, as EP3), `CUDA_RUNTIME`, `CUDA_DRIVER`, `GPU_FREE_BYTES` (`none` or 0 on the CPU).
- `COPY_TIME` per pass and in the summary (0 on the CPU).
- In the records too, so a GPU result says which GPU made it.

## 6. The checks

| Check | Compares | Expected |
|---|---|---|
| `--test-rng` on `star_gpu64` | cuRAND's first numbers of chunk 0 vs `rng.cuh` | identical |
| `--compare-gpu` (new) | `star_cpu64` vs `star_gpu64`, same photons | counts identical or nearly; images identical or a handful of photons apart (the GPU's math library and FMA) |
| `--check-passes`, `--check-split` with `--processor gpu` | as on the CPU | bit for bit |
| `--compare-ep2` with `--processor gpu` | EP2 vs the GPU | χ²/dof ≈ 1 |

## 7. The sweeps (today) [to review]

A **sweep** is a set of engine runs that vary **one thing**, to measure how the time responds. `sweeps.py` runs them; `scaling.py` analyses them.

### A point

**One point = one engine process**, run alone, from start to exit (EP3: experiments run independently).

| Fixed for every point | Value |
|---|---|
| The star | EP2's: a = 1, 3 spots (0.05 / 0.10), seed 42, chunk size 1024 |
| The image | 200², 150 frames, nmu 47, 4 observers, one pass (96 MB) |
| Recording | on (the atomics are part of the cost); the files are saved, then deleted; the summary is kept |

**What a point keeps** (its `SUMMARY`, as JSON): `TIME`, and the phases `SETUP` (includes creating the CUDA context), `ZERO`, `PHOTONS` (the kernel), `COPY`, `SAVE`; plus `CYCLES`, `SMS_USED`, `BLOCKS`, `THREADS_PER_BLOCK`, `REGISTERS_PER_THREAD`, and the process's wall time measured by Python (which adds the process start).

**Repeats:** each point runs 3 times; the analysis uses the median and shows the spread. Same seed, so the same photons: only the timing varies.

### The four sweeps

| Sweep | Law | Varies | Fixed | Answers |
|---|---|---|---|---|
| **`strong`** | Amdahl | SMs: 1, 2, 3, 4, 6, 9, 12, 18, 24, 36 | N | how much faster does *this* star get with more SMs? |
| **`weak`** | Gustafson | SMs (same list), and **N ∝ SMs** | photons per SM | can more SMs do *more* in the same time? |
| **`problem-size`** | Amdahl's p(N) | N: 10⁵ … 10¹⁰ (×10) | 1 SM and 36 SMs | how the parallel fraction grows with N (EP3's result, for the star) |
| **`devices`** | speedup | the binary: `star_cpu64`, `star_cpu32`, `star_gpu64`, `star_gpu32` | N | CPU vs GPU, FP64 vs FP32 |

- **`strong` and `weak` run twice: FP64 and FP32.** On the laptop FP64 is 1:64 of FP32, so the FP64 kernel is limited by arithmetic and the FP32 kernel more by the image's atomics: the two may scale differently.
- **N is set from a calibration**, not guessed: a short run at 36 SMs measures the speed, then
  - `strong`: N such that the 36-SM run's photon phase lasts **5 s** (so 1 SM lasts ~3 min);
  - `weak`: photons per SM such that each point lasts **5 s**;
  - `problem-size`: N up to what keeps the 1-SM run under ~3 min.
  The calibrated N goes into the sweep's description, so the sweep can be repeated exactly.
- **`devices`:** N = 10⁸ (the CPU takes ~10 s at that N; the GPU much less).

### Cost (estimate, before measuring the GPU)

| Sweep | Points × repeats | Time |
|---|---|---|
| `strong` FP64 + FP32 | 10 SM counts × 3, twice | the 36-SM time × Σ(36/n) ≈ 5 s × 82 ≈ 7 min per repeat → **~40 min** |
| `weak` FP64 + FP32 | 10 × 3, twice | 5 s each → **~5 min** |
| `problem-size` | 6 N × 2 SM counts × 3 | **~15 min** |
| `devices` | 4 × 3 | **~2 min** |

About an hour in total, running in the background while the rest of today's steps go on.

### How a sweep runs (the code of `sweeps.py`)

```python
def main():                                        # python sweeps.py strong --processor gpu --precision 64
    args = parse_args()
    binary = build_engine(args.engine)             # engine.py: nvcc, only if the code changed
    folder = new_sweep_folder(args) or the folder given to `continue`
    sweep = describe_sweep(args, binary)           # kind, units list, N (calibrated), repeats → sweep.json
    for point in points_of(sweep):                 # every (units, N, repeat), in an interleaved order
        if not point_is_done(folder, point):
            save_point(folder, point, run_point(binary, point))

def calibrate(binary, seconds):                    # N such that the run on all units lasts `seconds`
    result = run_engine(binary, photons=10**8, units=ALL)
    return round(seconds * result["PHOTONS_PER_SECOND"])

def points_of(sweep):                              # one function per kind
    strong:        [Point(units=n, photons=N)            for n in units]
    weak:          [Point(units=n, photons=n * N_per_unit) for n in units]
    problem-size:  [Point(units=n, photons=N)            for N in photons_list for n in (1, ALL)]
    devices:       [Point(binary=b, units=ALL, photons=N) for b in the four binaries]  (+ the CPU on 1 core)
    each list repeated `repeats` times: repeat 1 of every point, then repeat 2, ...

def run_point(binary, point):                      # one engine process
    started_at = perf_counter()
    summary = run_engine(binary, photons=point.photons, units=point.units, output_dir=a temporary folder)
    summary["PROCESS_TIME"] = perf_counter() - started_at
    delete the temporary folder (the image files)
    return summary

def save_point(folder, point, summary):            # point_sms09_n4.2e10_rep2.json, written .partial then renamed
```

- **One point is one call of the engine binary,** for example `build/star_gpu64 -n 4.2e10 --sms 9 -o /tmp/point`. The engine does the rest: it allows only 9 SMs (the mask), runs, writes the image, prints `SUMMARY`. Python keeps the `SUMMARY` and deletes the image.
- **Calibration first:** one short run on all units measures the speed; N follows from the 5 s target, and goes into `sweep.json`.
- **A warm-up run** before the first point, not saved: the GPU's clock ramps up from idle (825 MHz) and the first CUDA call pays for loading the driver.
- **Interleaved repeats:** repeat 1 of every point, then repeat 2, then 3, not three in a row. If the laptop heats and slows during the sweep, the slowdown spreads over all points instead of biasing the last ones (a rule from Hoefler and Belli, [hpc_architecture.md §4](hpc_architecture.md#4-benchmarking-like-hpc)).
- **Resuming:** `python sweeps.py continue <folder>` reads `sweep.json` and runs only the points without a file. A point file exists only when its run finished (`.partial`, then rename).
- **Python's own clock** (`PROCESS_TIME`) adds what the engine cannot see: starting the process and loading the CUDA driver.

### On disk

```
EP4/runs/<date>-sweep-strong-gpu64/
    sweep.json                         what was asked: kind, binary, SM list, N (calibrated), repeats
    point_sms09_n4.2e10_rep2.json      one point's SUMMARY + the process's wall time
```

- **A point that exists is skipped,** so an interrupted sweep continues where it stopped (the idempotence of [jobs_and_passes.md §5](jobs_and_passes.md#5-identity-and-resuming-running-a-configuration-is-idempotent), in miniature).

### What `scaling.py` makes from them

| Figure | From | Shows |
|---|---|---|
| **Strong scaling** | `strong` | speedup S(n) = T(1)/T(n) with the ideal line and the **Amdahl fit** T(n) = T(1)(s + p/n), for the **whole run** (`TIME`) and for **the kernel alone** (`PHOTONS`); efficiency S/n |
| **Weak scaling** | `weak` | time per point (ideal: flat), efficiency E(n) = T(1)/T(n), scaled speedup with the **Gustafson fit** S(n) = n − s′(n − 1) |
| **Strong vs weak** | both | the flattening curve and the straight line on one plot: the class's central figure |
| **Serial part, measured vs fitted** | `strong` | s from the fit vs (SETUP + ZERO + COPY + SAVE) / TIME measured directly |
| **p(N)** | `problem-size` | p from two points (1 and 36 SMs): p = (1 − 1/S)/(1 − 1/36), against N |
| **Devices** | `devices` | photons per second for the four binaries; the GPU's speedup over the CPU |

Each figure is saved with its data (`.json` next to the `.png`), so the slides can be redrawn without rerunning.

### Commands

```
python sweeps.py strong       --processor gpu --precision 64 --repeats 3
python sweeps.py weak         --processor gpu --precision 32 --repeats 3
python sweeps.py problem-size --processor gpu --precision 64
python sweeps.py devices
python scaling.py EP4/runs/<date>-sweep-strong-gpu64 EP4/runs/<date>-sweep-weak-gpu64 ...
```

## 8. What EP3 already settled (review of the EP3 sessions, 2026-11-21)

EP3 was built in other Claude Code sessions (on the Ubuntu instances, project `chat_buffon`). Their memory (the `ai-memory` store) keeps the prompts, not the answers; read together with `EP3/buffon.cu`, they show what was **tested and decided there**, so EP4 reuses it instead of redesigning it:

| Settled in EP3 | Where it shows | What EP4 does with it |
|---|---|---|
| **The work queue** (`atomicAdd` on `next_chunk`) with **persistent threads** (`take_chunks_until_none_left`) | `buffon.cu`, run for hours (the 1h40 delivery run) | **copied as is**, same names; only the chunk's body changes (photons instead of needles) |
| **The launch from the machine:** block size and blocks per SM from the occupancy API, grid = blocks per SM × SMs (EP3 prompt: "choose SM count based on the machine, as well as the threads") | `choose_launch` | the same, for `trace_photons` |
| **Scaling over SMs** with the `%smid` mask, after asking whether SMs are the right axis at all (blocks vs threads vs clusters) | `sm_id`, `allow_first_sms`, `list_sm_ids` | the same (`gpu_sms.cu`) |
| **Cooperative launch** when the GPU supports it, so all blocks are resident at once and the mask decides which SMs work ("we can launch kernels that won't run, they simply wait") | `run_kernel` | the same |
| **Seeds:** one Philox subsequence per chunk, not per SM | `count_crossings_in_chunk` | already EP4's rule |
| **Experiments run independently:** each point of a sweep is its own run | `buffon.py` | each sweep point is its own engine process (`sweeps.py`) |
| **Cycles** (`clock64`, the slowest block) next to the host's wall time | `GpuCounters.cycles` | **added:** `CYCLES` in EP4's GPU output |
| **Registers per thread** reported | `registers_per_thread` | **added:** `REGISTERS_PER_THREAD` (the star's kernel is much bigger than Buffon's, so occupancy may drop) |
| **A time budget** for the delivery run (default 10 min) | `buffon.py` | the sweeps take a budget too |
| **Adaptive chunk size** (bigger chunks for big N, a fix for N < 10⁷) | `choose_chunk_size` | **not taken, on purpose:** EP4's chunk size is fixed (1024) because the chunk *is* the random stream; a chunk size that changed with the launch would change the image. EP3 only counted crossings, so it could adapt |

**So the queue, the launch and the SM mask are not new design: they are EP3's, tested.** What is new in EP4's kernel is what happens inside a chunk (the star's physics and the image's atomics) and around it (passes, constant memory, copies back).

## 9. Decisions

- [x] **No Makefile:** `engine.py` builds (section 3), with the code hash deciding when to rebuild. *(decided 2026-11-21)*
- [x] **`GpuImage`** instead of the `DeviceImage` named in decision B: EP3 says `gpu` everywhere (`copy_from_gpu`, `GpuCounters`), and in CUDA "device" is the GPU only. *(agreed 2026-11-21)*
- [x] **Keep `rng.cuh`** and add `gpu_rng.cuh`. *(agreed)*
- [x] **The kernel uses EP3's queue from the start;** the SM mask in the next step. *(agreed; and EP3 already tested both, section 8)*
- [x] **A new phase, `COPY`** (GPU → host), measured apart from the disk. *(agreed)*
- [x] **Separate binaries** (`star_cpu64`, `star_gpu64`, …); CPU vs GPU compared by `star.py`. *(agreed)*
- [x] **Named by the hardware itself (option B):** `processor.cuh` (what the pass loop asks of a processor), `cpu.cu`, `gpu.cu`, next to `cpu_cores.cu`, `gpu_sms.cu`, `gpu_image.cu`, `gpu_device.cu`. Not "machine": that is the laptop or the A30, which both have a CPU and a GPU (the driver's `machines/`). Option A was by the role (`tracer.cuh`, `cpu_tracer.cu`, `gpu_tracer.cu`). *(decided 2026-11-21; the whole architecture follows the hardware: [hpc_architecture.md](hpc_architecture.md))*

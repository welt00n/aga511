# Nsight: looking inside the GPU

Material for the class. Two NVIDIA tools, two different questions:

| | **Nsight Systems** (`nsys`) | **Nsight Compute** (`ncu`) |
|---|---|---|
| Question | Where does the whole program's time go? | Why is this kernel fast or slow? |
| How | intercepts CUDA calls, timestamps them, samples the CPU | runs the kernel many times, reading a few hardware counters each time |
| Cost | a few % | 10–100× or more on the profiled kernel |
| Run with | the real problem size | a small N, one launch |
| Output | a timeline | sections per kernel |

Order of use: **our own timers** (cheap, always on) → **`nsys`** (confirms the timers, shows surprises) → **`ncu`** (only on the kernel that dominates the time).

---

## 1. How `nsys` works

```bash
nsys profile -o buffon_timeline --stats=true ./buffon -n 1e10
```

1. `nsys` **starts the program** as a child process and loads a library into it before `main`. The code is not changed or recompiled.
2. **It intercepts CUDA calls** through **CUPTI** (the CUDA Profiling Tools Interface). Every `cudaMalloc`, `cudaMemcpy` and kernel launch gets a start and end time. The GPU itself reports when each kernel and copy really started and ended.
3. **It samples the CPU**: it interrupts the program periodically and notes which function was running (statistical profiling).
4. **It traces the operating system**: threads, system calls (for example `ioctl`, which is how a program talks to the GPU driver).
5. **It writes a report** (`.nsys-rep`), opened in `nsys-ui` or in Nsight Systems on Windows. `--stats=true` also prints summary tables in the terminal.

## 2. Demonstration: Buffon (EP3) in `nsys`

RTX 5070 Laptop (36 SMs, CUDA 13.3), 10¹⁰ needles in FP64. Report: [`profiling/buffon_timeline.nsys-rep`](profiling/buffon_timeline.nsys-rep).

### 2.1 Is our timer right?

| Measurement | Value |
|---|---|
| EP3's own `LOOP_TIME` (CPU clock, around the kernel) | 1.4795 s |
| `throw_needles` according to `nsys` (GPU clock) | **1.4794 s** |

**Yes.** The difference is 0.1 ms in 1.48 s. EP3's cheap timer measures the kernel correctly.

### 2.2 Where is the serial part of Amdahl's law?

EP3's fit gave a serial part of **a ≈ 0.15 s**. The timeline shows where it comes from:

| Moment | Time (ms) |
|---|---|
| first traced CUDA call | 0 |
| first `cudaMalloc` | **154** |
| `throw_needles` starts | 157 |
| `throw_needles` ends | 1 636 |
| last CUDA call ends | 1 637 |

- **The 154 ms** between the first CUDA call and the first `cudaMalloc` are **CUDA context creation**: the driver preparing the GPU for the process. In that interval there are **406 `ioctl` calls** to the driver, adding up to 103 ms.
- **Everything else in the serial part is almost nothing:**
  - 6 `cudaMalloc`: 1.5 ms in total;
  - 5 `cudaMemcpy`: 0.4 ms;
  - the two kernel launches: 0.7 ms;
  - the `mark_sm_ids` kernel (which finds the SM ids): 0.8 µs;
  - the copies are tiny: a few KB.
- **Lesson:** Buffon's serial part is **the fixed cost of "switching on" the GPU**, not our code. It does not depend on N, which is why p → 1 as N grows.

### 2.3 What the profiler costs

| | without `nsys` | with `nsys` |
|---|---|---|
| `TIME` (total on the GPU) | 1.63 s | 2.06 s |
| `LOOP_TIME` (kernel) | 1.4836 s | 1.4795 s |

`nsys` adds ~0.4 s at the start (its own setup), but **the kernel does not change**. That is why `nsys` can measure the real program.

### 2.4 What to show live

Open `buffon_timeline.nsys-rep` in Nsight Systems and show:

1. the **CUDA API** row of the main thread: context creation, the mallocs, the long `cudaDeviceSynchronize` (the CPU waiting for the GPU);
2. the **CUDA HW** row: the `throw_needles` kernel as one 1.48 s bar, and `mark_sm_ids` almost invisible;
3. a zoom on the start: the 154 ms "gap" before the first malloc = the serial part of Amdahl's law.

---

## 3. How `ncu` works

```bash
ncu --set full -k throw_needles -c 1 -o buffon_kernel ./buffon -n 1e8
```

1. It also starts the program and intercepts kernel launches. `-k throw_needles -c 1`: only the first launch of that kernel.
2. The GPU has **hardware counters** (instructions issued, cache hits, DRAM bytes, why warps wait, ...), but it can only read **a few at a time**.
3. So `ncu` does **kernel replay**:
   - it saves the memory the kernel uses;
   - runs the kernel and reads one group of counters;
   - restores the memory and runs it again for the next group;
   - repeats this **dozens of times**, with the clocks locked so the passes are comparable.
4. It combines the passes into the report sections:
   - **Speed Of Light**: % of peak compute and of peak memory;
   - **Roofline**;
   - **Occupancy**: active warps per SM, and what limits them;
   - **Warp State Statistics**: why warps wait;
   - **Memory Workload Analysis**;
   - **Source**: counters per line (compile with `-lineinfo`).

**Times measured under `ncu` are not performance results**: `ncu` says *why*, not *how long*.

## 4. Demonstration: Buffon in `ncu`

### 4.1 First obstacle: permission

The first attempt failed:

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0.
```

Hardware counters are protected by default, because they can reveal what other programs do on the GPU. On WSL, permission is given **on Windows**:

1. open the **NVIDIA Control Panel** (or the NVIDIA App);
2. turn on **Desktop → Enable Developer Settings**, if it is still off;
3. **Developer → Manage GPU Performance Counters → "Allow access to the GPU performance counters to all users"**;
4. apply (it may ask for administrator rights);
5. restart WSL (`wsl --shutdown`) or reboot, so the setting reaches the driver. On our machine, after enabling it once, the driver still showed it unset (`RmProfilingAdminOnly` absent), so this step is still pending.

Worth mentioning in the class: it is the first error almost everyone hits with `ncu`.

### 4.2 Results

*(to fill in once the counters are enabled)*

What we expect to see, for the class to predict first:

- **Speed Of Light:** high compute, very low memory. The kernel barely touches memory: each thread accumulates in a register and does one `atomicAdd` at the end.
- **Bottleneck:** the FP64 units. On this GPU, FP64 runs at 1/64 of the FP32 rate (attribute `FP32_PER_FP64=64`).
- **Occupancy:**
  - 1 block of 1024 threads per SM;
  - 58 registers per thread (from `./buffon -i`);
  - the maximum is 1536 threads per SM, so the theoretical occupancy is 1024/1536 ≈ 67%.

---

## 5. Commands

```bash
# timeline, with tables in the terminal
nsys profile -o name --stats=true ./program args

# only the tables of a saved report
nsys stats name.nsys-rep

# one kernel in detail (small N!)
ncu --set full -k kernel_name -c 1 -o name ./program args

# ncu summary in the terminal
ncu --import name.ncu-rep --page details

# counters per line of code: compile with
nvcc -lineinfo ...
```

## 6. Lessons for the class

1. **Measure before optimizing**: `nsys` confirmed our timer and explained the serial part in one picture.
2. **Buffon's serial part is CUDA context creation** (~0.15 s): a fixed cost of the GPU, not of the algorithm.
3. **Profilers cost too**: `nsys` added 0.4 s to the total but did not touch the kernel. `ncu` changes the time a lot, so we never use its times.
4. **Hardware counters are protected** (`ERR_NVGPUCTRPERM`): they must be enabled in the driver.

# EP4 research log

**What this is.** The questions we asked of the engine and the hardware while building EP4, each answered by a measurement that can be repeated. Not a product: one standalone script per question in [`EP4/research/`](../EP4/research/), each printing what it measured and saving that output in `EP4/research/results/<script>.txt` (with the date, the machine, the engine's code hash and the nvcc build on top). The products (`main.py`) don't depend on any of it.

**How to run one** (from `EP4/`, in the `aga511` environment):

```bash
python research/register_liveness.py
```

Each script ends with what we found the first time ("Found on 2026-11-22: …"), so a re-run on another day or another machine can be compared with it at a glance. `research/common.py` holds what they share: compiling the kernel alone (cubin or PTX), reading its SASS and its register report, the SM's limits, the occupancy arithmetic, and the recorded output.

## The scripts

| Script | Question | Runs | Time |
|---|---|---|---|
| [`sm_limits.py`](../EP4/research/sm_limits.py) | what the SM can hold (registers, warps, blocks) | the GPU | seconds |
| [`fp64_in_fp32_kernel.py`](../EP4/research/fp64_in_fp32_kernel.py) | Q1: why the FP32 kernel has FP64 instructions | compile only | ~30 s |
| [`nop_runs.py`](../EP4/research/nop_runs.py) | Q2: what the FP64 kernel's 1063 `NOP`s are | compile only | ~1 min |
| [`pass_order.py`](../EP4/research/pass_order.py) | Q3: is the second pass slower? | GPU FP64 | ~1 min |
| [`cpu_baseline.py`](../EP4/research/cpu_baseline.py) | Q4: how much the one-core CPU baseline moves, and why | CPU (and the GPU once) | ~4 min |
| [`clock_reasons.py`](../EP4/research/clock_reasons.py) | why the GPU's clock falls | GPU | ~1 min |
| [`register_sweep.py`](../EP4/research/register_sweep.py) | registers → occupancy → speed, with `-maxrregcount` caps | GPU, 13 builds | ~10 min |
| [`a30_registers.py`](../EP4/research/a30_registers.py) | how many registers the kernel will use on the A30 | compile only | ~1 min |
| [`front_end_registers.py`](../EP4/research/front_end_registers.py) | why sm_80 needs more registers than sm_120 | compile only | ~3 min |
| [`register_liveness.py`](../EP4/research/register_liveness.py) | where in the code the registers go | compile only | ~1 min |
| [`a30_prediction.py`](../EP4/research/a30_prediction.py) | how fast the A30 should be, before measuring it | compile only | ~1 min |
| [`register_cap_instructions.py`](../EP4/research/register_cap_instructions.py) | why capping registers for occupancy is slower: the code it makes | compile only | ~2 min |
| [`binning_costs.py`](../EP4/research/binning_costs.py) | how much the bins (frames, nμ) blur the image, and what sharpness costs | nothing (a calculation) | seconds |
| [`bit_for_bit_reference.py`](../EP4/research/bit_for_bit_reference.py) | a tool, not a question: `save` before a refactor, `compare` after | all 4 binaries | ~1 min |

## The entries

### 2026-11-22 · Q1: FP64 instructions in the FP32 kernel · [`fp64_in_fp32_kernel.py`](../EP4/research/fp64_in_fp32_kernel.py)
- **Question.** The instruction mix found FP64 instructions in the FP32 kernel. Is a `double` leaking in somewhere?
- **How.** `ptxas --warn-on-double-precision-use`, the FP64 instructions in the SASS, and the `.f64` operations in the PTX mapped to source lines through `-lineinfo` (`.loc` with `inlined_at`).
- **Result.** 12 SASS instructions, 3 per call (`I2F.F64.S64` → `DMUL` by a constant → `F2F.F32.F64`), in each of the 4 `sinf`/`cosf` calls: real.cuh:26/28, used at photon.cuh:36–37 (the surface point) and 56–57 (the local azimuth).
- **Meaning.** This is CUDA's own Payne–Hanek argument reduction: it multiplies by 2/π in high precision, and only runs for |x| > ~10⁵ rad. Our angles are below 2π, so it never runs. No leak, and nothing to fix.

### 2026-11-22 · Q2: the `NOP`s · [`nop_runs.py`](../EP4/research/nop_runs.py)
- **Question.** The FP64 kernel has 1063 `NOP`s (39% of its 2744 instructions). Are they alignment padding that never runs?
- **How.** The runs of consecutive `NOP`s in the SASS: their lengths, and the instructions before and after them.
- **Result.** 304 runs, 233 of them exactly 4 long, mostly right before a `DFMA`, `DMUL` or `DADD` that needs the previous FP64 result. Only 9 are padding, at the end. FP32 has 87 `NOP`s, mostly single ones before a branch's reconvergence point.
- **Meaning.** The guess was wrong. They are waits: this GPU has few FP64 units, the compiler knows their latency, and it makes the warp wait. They are executed. This is also why the instruction mix is *static*: it counts instructions in the code, not instructions executed.

### 2026-11-22 · Q3: the slow second pass · [`pass_order.py`](../EP4/research/pass_order.py)
- **Question.** Once, pass 2 ran 5× slower after a 12 s save. Did the clock drop while the GPU idled?
- **How.** Two passes of 4 observers, both orders (equator side first, pole side first), N = 10⁷ and 10⁸, 3 repeats. For each pass: the kernel time and the cycles of the slowest SM. The same cycles at a different time would mean the clock moved.
- **Result.** Not reproduced. A pass right after the clock drop was the fastest. At 10⁷ (kernels of 25–44 ms) pass 2 / pass 1 ranged from 0.63 to 1.60, in either order; at 10⁸ (~157 ms) from 0.99 to 1.00. In the re-run, the slow kernels also took more **cycles** on the slowest SM, in proportion: the clock was the same, the work wasn't.
- **Meaning.** At 10⁷ there are 9 766 chunks of 1024 photons for 27 648 resident threads (36 SMs × 768, the launch the occupancy API picks), so a thread gets one chunk or none, and the slowest SM decides the time. That is the reason for the minimum N in the products: at least 32 chunks per resident thread (and 1 s of work), so every SM has its fair share.

### 2026-11-22 · Q4: the CPU baseline · [`cpu_baseline.py`](../EP4/research/cpu_baseline.py)
- **Question.** The CPU's one-core FP64 speed was 7.0–8.0·10⁶ photons/s during the sweeps and 8.8–9.1·10⁶ in every re-run. Does the GPU take the laptop's power?
- **How.** Pinned to each logical CPU (`taskset`); not pinned, 8 runs; alone vs with `nvidia-smi` sampling; before vs right after a long GPU run.
- **Result (first run, morning).** 8.8–9.1·10⁶ in every case: not the core, the run length, the GPU's heat or the monitor.
- **Result (the script's re-run, afternoon).** Wider: pinned, 7.6·10⁶ (CPU 0) to 9.4·10⁶ (CPUs 17–18), rising roughly with the CPU's number; not pinned, 8.0–8.9·10⁶; the monitor changes nothing (ratio 1.01); right after 36 s of GPU FP32, 0.98×. The spread between cores and between runs (~20%) is as large as the sweep's gap, so the sweep's 7.0–8.0·10⁶ is within the CPU's normal variation, not a GPU effect.
- **Meaning.** The shared-power guess isn't supported. The baseline varies ~25% between sessions, so the speedups against the CPU are given with the CPU's rate measured on the same day ("82×, ~65× on other days"). The CPU is a reference only; the GPU's own scaling doesn't depend on it.

### 2026-11-22 · Why the clock falls · [`clock_reasons.py`](../EP4/research/clock_reasons.py)
- **Question.** The clock drops from ~2.88 GHz on 1 SM to ~1.95 GHz on 36. Is it the register spills (they appeared at the same time), or power?
- **How.** `nvidia-smi`'s `clocks_event_reasons.*` (the driver's own reasons for holding the clock down), sampled every 250 ms during three runs: FP32 on 36 SMs, FP64 on 36 SMs, FP32 on 1 SM.
- **Result.** FP64, and FP32 on 1 SM, run at 2880–2895 MHz, held by `reliability` (the voltage ceiling). FP32 on 36 SMs is held by `sw_power_cap`: the enforced limit moves between 85 and 115 W (the default is 80 W and the maximum 115 W, likely Dynamic Boost), and the clock follows it. In the re-run: 2020 MHz in the first sample (133 W drawn), then 2620–2690 MHz at the 115 W limit. FP64 on 36 SMs draws only ~57 W.
- **Meaning.** It's power, not spills: FP64 keeps the clock because its few FP64 units draw less. We don't lock clocks or change power limits (system settings); instead, the speed is also measured in cycles, which don't depend on the clock.

### 2026-11-22 · The SM's limits · [`sm_limits.py`](../EP4/research/sm_limits.py)
- **Result (RTX 5070 Laptop, cc 12.0).** 36 SMs; per SM: 65 536 registers, 1536 threads (48 warps), 24 blocks, 102 400 B of shared memory; warps of 32. Registers are allocated 8 per thread at a time (256 per warp).
- **The A30 (cc 8.0)** can't be queried from here. From NVIDIA's Ampere tuning guide: 56 SMs; 65 536 registers, 2048 threads (64 warps), 32 blocks per SM.

### 2026-11-22 · Register sweep · [`register_sweep.py`](../EP4/research/register_sweep.py)
- **Question.** The registers limit occupancy (80 registers → 0.50 in FP64). Would more warps per SM be faster?
- **How.** Cap the registers with `-maxrregcount` (FP64: none, 72, 64… 32; FP32: none, 64… 32) and measure the launch's occupancy, the spills, the kernel time, the cycles, the clock and whether the image is the same. The first run (scratch, kept in `runs/kernel/2026-11-22_12-02-39-registers-sweep`) ran the caps in order; the script now interleaves them (forward, then backward) and gives the speed in cycles too.
- **Result (first run).** FP64: occupancy follows the register limit exactly; 80 → 0.50 and 72 → 0.58 run at the same speed; 40 → 1.00 is 9% slower, with 344 B spilled. Images identical at every cap. FP32 was inconclusive: the clock fell from 2748 to 2017 MHz during the sweep.
- **Result (interleaved run, N = 10⁹ FP64 / 10¹⁰ FP32, 3 repeats).** The clock stayed steady this time (FP64 2881–2886 MHz, FP32 2700–2730, except 2605 at cap 32), and cycles and time agree:

  | FP64 registers | 80 | 72 | 64 | 56 | 48 | 40 | 32 |
  |---|---|---|---|---|---|---|---|
  | occupancy | 0.50 | 0.58 | 0.67 | 0.75 | 0.83 | 1.00 | 1.00 |
  | spilled (B) | 0 | 0 | 24 | 84 | 164 | 344 | 444 |
  | cycles vs uncapped | 1.000 | 0.995 | 0.998 | 1.049 | 1.068 | 1.094 | 1.096 |

  | FP32 registers | 67 | 55 | 52 | 48 | 40 | 32 |
  |---|---|---|---|---|---|---|
  | occupancy | 0.58 | 0.75 | 0.75 | 0.83 | 1.00 | 1.00 |
  | spilled (B) | 0 | 0 | 0 | 24 | 80 | 156 |
  | cycles vs uncapped | 1.000 | 1.012 | 1.031 | 1.037 | 1.038 | 1.049 |

  Images identical at every cap. (FP32's caps of 64 and 56 give 55 and 52 registers: the compiler goes below the cap.)
- **Meaning.** On the laptop, more occupancy never pays. FP64 is bound by its few FP64 units, not by latency: up to 0.67 it's free (0.5% faster at 72), beyond that the spills cost up to 10%. FP32 is fastest uncapped: even without spills (55 registers, 0.75), it's 1% slower, because the compiler has to recompute values or schedule worse to fit. The register limit isn't hiding speed here. On the A30 (fast FP64, 96 registers, occupancy 0.33) the answer may differ: run this script there on Monday.

### 2026-11-22 · Registers on the A30 · [`a30_registers.py`](../EP4/research/a30_registers.py)
- **How.** Compile only, for sm_80, and read `ptxas -v`; then the occupancy arithmetic with the A30's limits.
- **Result.** FP64 96 registers, FP32 76, no spills (against 80 / 67 on sm_120): 21 / 25 warps of 64, so occupancy at most 0.33 / 0.39 (672 / 800 threads per SM).

### 2026-11-22 · Why sm_80 needs more registers · [`front_end_registers.py`](../EP4/research/front_end_registers.py)
- **Question.** Same code, same compiler: why 96 registers on sm_80 and 80 on sm_120?
- **How.** Separate the two halves of the compiler. The front end (NVVM) turns C++ into PTX for a `compute_XX`; the back end (`ptxas`) turns PTX into SASS for an `sm_XX`. Make the PTX for each compute_XX and assemble it both for sm_120 and for its own SM.
- **Result.** The jump is in the front end, exactly at compute_100:

  | PTX made for | FP64 on sm_120 | FP64 on its own SM | FP32 on sm_120 | FP32 on its own SM | FP64 PTX branches |
  |---|---|---|---|---|---|
  | compute_75 … 90 | 98 | 96 | 71 | 72–76 | 107 |
  | compute_100 … 110 | 80 | 84 | 67 | 70 | 90 |
  | compute_120 | 80 | 80 | 67 | 67 | 90 |

  Assembled for the same SM, the older PTX needs 18 more registers in FP64. `ptxas` adds only 2–5 between SMs.
- **Meaning.** CUDA 13 compiles for Blackwell and later (compute_100+) with a newer NVVM (based on LLVM 18); older targets keep the older one. Its PTX has fewer branches. The newer front end can't be asked for from `nvcc`: `-nvvm-latest` is a libNVVM/nvJitLink option, and nvcc rejects it.

### 2026-11-22 · Where the registers go (liveness) · [`register_liveness.py`](../EP4/research/register_liveness.py)
- **Question.** To lower the registers, would we have to change the physics? Is it the number of variables?
- **How.** `nvdisasm -plr -lrm count` prints, for every instruction, how many registers are live at that moment. `nvdisasm -g -gi` gives each instruction's source line and the lines it was inlined at. Joined by address.
- **Result.** FP64 peaks at 74 live (80 allocated), FP32 at 63 (67). Both peaks are inside `sin`/`cos` (real.cuh:27/29 in FP64, 26/28 in FP32) called for the local azimuth in `sample_emission_direction` (photon.cuh:56–57, from photon.cuh:124). Elsewhere: the photon's physics has a median of 59 live in FP64 (54 in FP32); Philox 31–65.
- **Meaning.** A register is held only while its value is live, and values that never overlap share one. The count is the **peak** of simultaneously live values (doubles and 64-bit integers take 2), not the number of variables. At the peak, all of these are live together: the surface point (kept for the spot test and the pixel), the direction's ingredients, Philox's state, the loop's indices, the five 64-bit tallies, and sin/cos's own temporaries. Ways down without changing the results: `sincos` instead of separate `sin` and `cos`, 32-bit counters inside a chunk, and one tally fewer (missed = simulated − blocked − recorded). None of it was done: the laptop wouldn't gain (see the register sweep), and the A30 sweep decides first.

### 2026-11-22 · The bins blur the image · [`binning_costs.py`](../EP4/research/binning_costs.py)
- **Question.** Are the images as sharp as their pixels? How do the observers and the bins (`nphi` = frames, `nμ`) interact?
- **How.** The engine ([photon.cuh](../EP4/engine/physics/photon.cuh), `observer_bin` and `pixel_index`): a photon's direction picks its frame (azimuth, 2π/frames per bin) and its μ bin (2/nμ per bin), and its pixel is its surface point **projected along its own direction**. So a frame blends every view inside its cell. The blur is about R·2π/frames along the rotation and R·Δi across it, with Δi the bin's width in inclination (R = size/2 pixels).
- **Result (1024², 300 frames, nμ 95).** About 11 px along the rotation and at the equator (the rule nμ = frames/π balances the two there); 21 px at 30°, 12.5 px at 60°; **the polar bin spans 0–11.8°: up to 105 px**. Equal-μ bins hold equal photons but not equal angles: Δi ≈ (2/nμ)/sin i, and the polar bin is ≈ 2/√nμ wide.
- **Meaning.** Two axes with two costs. *Resolution* (frames, nμ) sets the blur and the photons: N = SNR²·nμ·frames·(π/4)·size²·(1 + 2a/3), so halving the blur costs 4× the photons, and keeping the blur in pixels while doubling the size costs 16×. *Image size* (pixels × frames × observers × 4 bytes) sets the memory, and past the GPU's memory the passes, each of which traces all N again. The **observers** cost no photons (each keeps its own bin's photons), only memory. A 1k image at 300 frames has about 100 independent elements across the disk; pixel-sharp images (frames ≈ π·size, nμ ≈ size) are out of reach with EP2's algorithm. "Peel-off" (every photon added to every observer and frame with its exact geometry) would remove the blur and the 1/(nμ·frames) dilution, but it is another algorithm (for the TCD). In the products: `measure`'s **costs** section (report section 6).

### 2026-11-22 · Why more occupancy is slower here · [`register_cap_instructions.py`](../EP4/research/register_cap_instructions.py)
- **Question.** The register sweep says that capping the registers to raise the occupancy never paid on the laptop. But a spill just goes to local memory: why would that be slower? Does synchronization get messy?
- **How.** For each `-maxrregcount` cap, the kernel's machine instructions (static), the spill stores and loads (`STL`, `LDL`), and the compiler's registers and spilled bytes; next to the register sweep's cycles.
- **Result.**

  | FP64 cap | registers | spilled | instructions | STL / LDL | cycles (sweep) | occupancy |
  |---|---|---|---|---|---|---|
  | none | 80 | 0 B | 1681 (1.000×) | 2 / 3 | 1.000 | 0.50 |
  | 72 | 72 | 0 B | 1.021× | 2 / 3 | 0.992–0.995 | 0.58 |
  | 64 | 64 | 24 B | 1.055× | 5 / 5 | 0.998 | 0.67 |
  | 40 | 40 | 344 B | 1.153× | 68 / 48 | 1.09 | 1.00 |

  In FP32, 55 registers with **0 B spilled** already has 4.9% more instructions and costs 1.2% more cycles; 32 registers: +11% instructions, 156 B spilled. (The few `STL`/`LDL` in the uncapped builds are the thread's small local arrays, not spills.)
- **Meaning.** Occupancy itself costs nothing; **taking registers away does**, in two steps. (1) Before any spill, the compiler keeps fewer values alive: it recomputes some instead of keeping them, and it can overlap fewer independent instructions. (2) Then it spills: each spilled value is a store and a later load. "Local" memory is private to the thread but lives in device memory, cached by L1 and L2: an L1 hit costs ~30 cycles, an L2 hit hundreds. And L1 is shared by every warp on the SM, so more warps × more spilled bytes means more traffic in the same cache: at 40 registers, 344 B × 1536 threads ≈ 530 KB of spills per SM, more than its L1. More warps would only pay back if the SM had idle units waiting on latency, and it doesn't here: in FP64 every warp queues for the few FP64 units (1:64), and in FP32 28 warps already keep the units busy. No synchronization is involved: the kernel has no `__syncthreads`, and spills are per thread. On the A30 (fast FP64, 96 registers, occupancy 0.33) the balance may differ: the register sweep there will say.

### 2026-11-23 · The A30, predicted · [`a30_prediction.py`](../EP4/research/a30_prediction.py)
- **Question.** How fast should the A30 be, in FP32 and FP64, before we measure it at IAG? (For the class: predict, then measure.)
- **How.** The laptop's 1k measurement gives the photons per clock cycle per SM (FP32 0.109, FP64 0.0064). The A30's SM (datasheet) has half the FP32 lanes but a separate INT32 pipe, and 16× the FP64 units; ~1.41 GHz, 56 SMs. FP32 per cycle per SM: between 0.5× (everything was lane-bound) and 1× (the separate integer pipe compensates). FP64: no longer bound by its FP64 units, so 0.6–0.85× the A30's FP32. The registers and occupancy come from compiling for sm_80.
- **Result (predicted).** FP32 4.3–8.6·10⁹ photons/s on 56 SMs (0.4–0.8× the laptop); FP64 2.6–7.3·10⁹ (4–11× the laptop). 96 / 76 registers, occupancy ≤ 0.33 / 0.39; minimum N ≈ 2.6·10⁹ (FP64), 4.3·10⁹ (FP32). The 2k showcase at SNR 10 (4 observers, one pass in 24 GB): 31–61 min.
- **Meaning.** On the A30 the precision trade-off flips: FP64 costs about as much as FP32, where on the laptop it is 17× slower. The range is wide because this kernel is mostly integer and FP32 work, not one unit's peak; the measurement at IAG narrows it to a number.

# CUDA class: plan

Working plan for the CUDA class. It changes as we learn; the concepts are collected in [`../docs/README.md`](../docs/README.md).

**Language: all docs, including this plan, are in English. Only the class itself and its slides will be in Portuguese.**

---

## Context

- **Who teaches:** a physics student at USP, in the CUDA team of AGA 511. Five years teaching English: comfortable at the board and with a room, new to slide presentations.
- **Who listens:** the AGA 511 class (Métodos Computacionais em Astronomia, IAG/USP, 2nd semester 2026) and **Prof. Alex Carciofi**, who works in HPC for astronomy (Monte Carlo radiative transfer). The students already know Amdahl's law, Monte Carlo and Buffon's needle (EP1, EP3); the other teams cover MPI, OpenMP and OpenACC.
- **Level:** high-level but precise. A professor of HPC is in the room, so every claim must be backed by the CUDA documentation or a measurement on our own GPU.
- **What it is:** **a class, not a presentation.** The board carries the reasoning (diagrams, derivations, predictions); slides only carry what cannot be drawn well (code, plots, timelines, photos).
- **Our material:**
  - EP3: Buffon on the GPU, with an Amdahl study over SMs.
  - EP4: the star, in progress.
  - The hardware: an RTX 5070 Laptop GPU, 36 SMs, CUDA 13.3.
  - The profilers: Nsight Systems and Nsight Compute.

## The professor's syllabus (whiteboard, read 2026-11-23)

Written for two presenters; now one. In the original Portuguese headings:

1. **História** (5 min): problemas, evolução, ciência.
2. **Arquitetura** (divisão): multiprocessadores; arquitetura antiga vs atual; núcleos FP32, FP64, INT32, tensor; precisão numérica.
3. **Terminologia básica:** host/device, thread, bloco, grid → modelo de memória.
4. **Fluxo de execução:** transferência de dados, lançamento do kernel, execução, retorno → latência; CPU (flops) vs memória.
5. **Modelo de programação:** SIMD/SIMT; exemplo prático, *redução paralela* (distribuição, resultados parciais, combinação, resultado final); cooperação entre threads, sincronização, shared memory, limitações; comparação.
6. **Buffon:** EP3 → Amdahl; comparações.
7. **Plano de aula** (for each project): intro → detalhes do código serial → paralelização → Amdahl e discussão. Sketch: a fork-join (S, then P P P P, then the join).

## The sequence (draft, 2026-11-23)

The projects drive the class; the foundations come first (the syllabus' 2–5), then EP3, then the harder concepts, then the star. Each EP follows the board's four steps (intro → serial code → parallelization → Amdahl). **EP3 stays as delivered:** only its Amdahl analysis. **The A30 enters as a prediction** (measured at IAG on Monday 2026-11-23; the slides switch to the measurement if it is ready).

| # | min | Block | Syllabus | Content |
|---|---|---|---|---|
| 0 | 7 | Prelude: history | 1 | the problems GPUs were made for; graphics → science; GPUs in science and industry today |
| 1 | 3 | The hook | — | the star's video: 3.9·10¹² photons, 6 min on this GPU, 3.5 h on one of its SMs |
| 2 | 10 | Architecture | 2 | CPU vs GPU (latency vs throughput); the SM and its units (FP32, FP64, INT32, tensor); precision |
| 3 | 10 | Terminology and memory | 3 | host/device, thread → block → grid, warps; registers, shared, global, constant |
| 4 | 8 | Execution flow | 4 | copy in → launch → execute → copy back; latency; flops vs memory |
| 5 | 17 | Programming model | 5 | SIMD vs SIMT; parallel reduction as the practical example; cooperation, synchronization, shared memory, limitations |
| 6 | 15 | EP3: Buffon | 6, 7 | intro → serial code → parallelization → Amdahl over SMs (as delivered) |
| 7 | 12 | Going deeper | — | warps and latency hiding; registers → occupancy (and why more is not faster here); FP64 vs FP32 |
| 8 | 15 | EP4: the star, the code | 7 | the physics → EP2's serial algorithm → chunks with their own random streams, atomics on the image, constant memory, GPU = CPU bit for bit |
| 9 | 12 | EP4: Amdahl and discussion | 7 | photons 36× vs whole run 19× at 1k: the image is the serial part; the curve toward ideal at large N; fair benchmarks; **the A30: predicted, then measured** |
| 10 | 6 | Result and closing | — | the SNR ladder; three takeaways |
| | 5 | Questions | | |

## The deck's sequence (2026-11-23, replaces the table above)

The slides are [`deck/aula.html`](deck/aula.html) (reveal.js, presented with [`deck/apresentar.py`](deck/README.md)). Every data point in them comes from the orchestrator; the map is [class_points.md](class_points.md). 77 slides, 11 parts, each opened by a divider with the agenda.

| Part | min | Content | New since the first draft |
|---|---|---|---|
| opening | 3 | title; agenda; **the thread** (concept → history → foundation → project) | the thread table |
| 1. History | 10 | science → industry → together (timeline); one era per slide, each leaving its concept | the timeline diagram, the era strip |
| hook | 2 | the star's video, "how?" | |
| 2. Architecture | 8 | CPU vs GPU (latency vs throughput); inside an SM; our two GPUs; FP32 vs FP64 bits | three diagrams |
| 3. Terminology and memory | 8 | host and device; a minimal kernel and its index; thread → warp → block → grid → SMs; the memory model | four diagrams |
| 4. Execution flow | 6 | the four arrows; what each costs (`phases_fp64.png`) | the phases figure from `measure` |
| 5. Programming model | 13 | SIMT and divergence; the reduction tree and its code; races and atomics; bridge to EP3 | three diagrams |
| 6. Random numbers | 12 | how many numbers; stateful (LCG) vs counter-based (Philox); a Philox round; a stream per chunk; **the six levels of validation**; the real `--test-rng` output; uniforms in (0, 1]; the χ² figure | **the whole part** |
| 7. EP3 | 13 | the geometry; the serial code; the error falls as 1/√N; persistent threads and a queue; the GPU code; Amdahl explained; Amdahl over SMs | the geometry, queue and Amdahl diagrams |
| 8. Deeper | 12 | latency hiding; registers → warps (drawn); the launch line as printed; the register sweep; why (instructions per cap); FP64 16× not 64×; the A30 | instructions per cap from `measure` |
| 9. EP4 | 14 | the star; the life of a photon (with each step's share of the time); the skeleton; where each thing lives; validation back to the pyramid; the two axes of cost | four diagrams |
| 10. EP4 Amdahl | 12 | where the time goes; the ceiling (linear, cycles, clock); large N; EP3 vs EP4; the laptop vs the A30; fair comparison | |
| 11. Closing | 5 | the SNR ladder; the thread, back; three lessons; the four commands | |
| questions | 2 | | |

## Block by block

For each block: the **message** (what the room must leave with, also the slide titles' voice), the **board**, the **slides** (only what can't be drawn), the **numbers**, and a **checkpoint** (a question to the room). **⟨A30⟩** marks a space filled after the IAG run on Monday: the class is written with the comparison in it. Figures: `M/` = `EP4/runs/measure/2026-11-22_20-25-24-Welton-all/figures/`, `S/` = `EP4/runs/make/2026-11-22_20-07-02-1024-f300-gpu32-snr10/`, `E3/` = `EP3/runs/2026-11-08_20-36-22-gpu-fp64/`. The class notes (a text to study from, one section per block) come after this plan.

### 0. History and usage → the concepts (18 min) *(replaces the separate history and architecture blocks, 2026-11-23)*
- **Message:** every part of a GPU answers a problem someone had; the history explains the architecture.
- **Board:** a timeline; at each era, the concept it left is written under it, and the SM grows on the side of the board as the eras add units.

  | Era | The problem, the users | The concept it leaves |
  |---|---|---|
  | late 1990s, fixed pipelines (GeForce 256, 1999) | games: millions of independent pixels per frame | data parallelism; throughput vs latency; many simple units |
  | early 2000s, programmable shaders (GeForce 3, 2001) | a small program per pixel | the kernel |
  | 2003–2006, GPGPU through graphics (BrookGPU, 2004) | scientists disguising matrices as textures | why a general programming model was needed |
  | 2006–2007, unified shaders and CUDA (G80) | general-purpose computing in C | host/device, the launch, the SM of identical cores |
  | ~2008–2010, FP64, ECC, caches (Tesla, Fermi) | science: double precision, reliable memory | numerical precision (FP32 vs FP64 units, 1:2 vs 1:64); the memory hierarchy |
  | 2012, deep learning (AlexNet, two GTX 580) | neural networks are matrix products | throughput computing beyond graphics |
  | 2017, tensor cores (Volta) | AI training | specialized units: FP32, INT32, FP64, tensor in one SM |
  | 2020s, exascale and AI (Frontier, 2022) | energy per flop | power limits: why the clock falls (our laptop's power cap) |
- **Usage:** graphics; astronomy (Monte Carlo radiative transfer, N-body); climate; molecular dynamics; AI.
- **Slides:** one per era at most (a picture, a number); then our two GPUs side by side (36 SMs, 128 FP32/INT32 lanes and 2 FP64 per SM, 1:64 | 56 SMs, 64 FP32 + 64 INT32 + 32 FP64 per SM, 1:2).
- **Precision:** FP32 has 24 bits of mantissa, FP64 53; in the star FP32 moves 7·10⁻⁴ of the photons, below the noise.
- **Checkpoint:** "What does drawing a frame have in common with throwing 10¹⁰ needles?"; "Which of our two GPUs wins in FP64?" (answered in block 9 with ⟨A30⟩).

### 1. The hook (3 min)
- **Message:** a picture made of 3.9·10¹² photons, in 6 minutes on a laptop.
- **Slides:** `S/observers_around.mp4` (7 observers, both hemispheres, 1k): play one loop.
- **Numbers:** N = 3.9·10¹² for SNR 10; 6.3 min on the GPU's 36 SMs, 3.5 h on one of them. ⟨A30: the 2k version, N = 1.6·10¹³, time⟩.
- **Checkpoint:** "How?" (left open; answered by block 9.)

### 3. Terminology and memory (10 min)
- **Message:** host and device are two computers with two memories; the work is a grid of blocks of threads, run 32 at a time (a warp).
- **Board:** the hierarchy, built step by step (thread → warp → block → grid → SMs); the memory spaces next to it (registers per thread, shared per block, global for all, constant broadcast to all).
- **Numbers:** our GPU holds 48 warps (1536 threads) per SM and 65 536 registers per SM.
- **Checkpoint:** "If a block has 768 threads, how many warps is that?"

### 4. Execution flow (8 min)
- **Message:** copy in → launch → run → copy back, and every arrow costs time.
- **Board:** the four arrows on a time axis, with what each costs.
- **Numbers (measured):** creating the CUDA context ~0.15 s (EP3, seen in `nsys`); copying a 5 GB image back ~0.4 s (~13 GB/s); saving it to disk ~4.5 s. Flops are cheap, moving data is not.
- **Slides:** the `nsys` timeline of Buffon ([nsight.md](nsight.md)).
- **Checkpoint:** "Where would you expect the serial part of a GPU program to hide?"

### 5. Programming model (17 min)
- **Message:** one instruction, 32 threads (SIMT); threads cooperate inside a block (shared memory, `__syncthreads`), and across blocks only through global memory and atomics.
- **Board:** SIMD vs SIMT, and divergence (an `if` splits a warp).
- **The practical example, a parallel reduction** (sum of N numbers): distribute (each thread its share), partial results (in registers), combine (a tree in shared memory, one `__syncthreads` per level), the final result (one atomic per block, or a second kernel). Its limits: synchronization only inside a block; shared memory is small; the tree's last levels leave threads idle.
- **Slides:** the reduction kernel (~15 lines); a figure of the tree.
- **Comparison:** serial N steps; parallel N/p + log p: a first glimpse of Amdahl.
- **Checkpoint:** "Buffon counts the needles that cross a line. What kind of operation is that?" (a reduction: the bridge to block 6)

### 6. EP3: Buffon (15 min), as delivered
- **Board's four steps:** intro (needles, π) → the serial code (one loop) → the parallelization (each thread its own random stream, counting in registers, one atomic at the end: block 5's reduction; persistent threads taking work from a queue) → **Amdahl over SMs**.
- **Slides:** `E3/amdahl.png` (the speedup over SMs and p vs N).
- **Numbers:** at N = 3.2·10¹⁰ (FP64): **34.3× on 36 SMs, p = 0.9986**; the serial part is 0.15 s (the CUDA context); the needle loop alone p = 0.9997. p grows with N: 0.07 at 10⁶, 0.78 at 10⁸, 0.997 at 10¹⁰. π to 14 decimals would take 2.7·10¹¹ years on this GPU.
- **Checkpoint (predict first):** "On 36 SMs, how much faster?"

### 7. Going deeper: warps, registers, occupancy (12 min)
- **Message:** an SM hides latency by switching between warps, and how many warps fit is decided by the registers.
- **Board:** 65 536 registers per SM ÷ (registers per thread × 32) = warps that fit. FP64: 80 registers → 25 warps allowed, 24 launched (whole blocks), occupancy 0.50; FP32: 67 → 28 warps, 0.58.
- **Slides:** `M/registers_and_occupancy.png`.
- **The result:** capping the registers raises the occupancy but is not faster: the compiler adds instructions (+5% before any spill), then spills to local memory (which lives in DRAM). FP64 is 17× slower than FP32 here, not 64×: the warps wait on the few FP64 units (the `NOP`s in the code are those waits).
- **⟨A30:⟩** 96 registers (FP64) → 21 of 64 warps, occupancy ⟨measured⟩; does more occupancy help there ⟨register sweep⟩?
- **Checkpoint:** "If we halve the registers per thread, do we go twice as fast?"

### 8. EP4: the star, the code (15 min)
- **Board's four steps:** intro (a spotted, limb-darkened star seen by observers at different inclinations; each frame is one phase of the rotation) → the serial code (EP2's algorithm: a photon leaves the surface, maybe hits a spot, escapes in a direction that picks its frame and observer, lands in a pixel) → the parallelization: Buffon's skeleton with a new body: chunks with their own random streams (the same numbers on CPU and GPU), one photon function compiled for both (`__host__ __device__`), atomic adds into a large image in global memory, the star's description in constant memory → correctness.
- **Numbers:** **the GPU's image equals the CPU's, bit for bit (FP64)**; FP32 moves 7·10⁻⁴ of the photons (below the noise). Statistically: GPU vs CPU with independent photons, χ²/dof = 0.99.
- **Slides:** the kernel's loop (~10 lines); one still frame.
- **Checkpoint:** "Why can two different processors give the *same bits*?"

### 9. EP4: Amdahl and discussion (12 min)
- **Message:** the same law as Buffon, a different serial part: for a high-resolution picture, the image itself.
- **Slides:** `M/amdahl_ceiling_fp64.png`; the phase table (FP64, N = 4·10⁹): photons 216.9 s → 6.0 s on 36 SMs (36×), setup + copy + save 5.6 s → 5.6 s, the whole run 223.5 s → 11.6 s (**19×**, p = 0.974, ceiling ~40). Then `M/costs_large_n.png`: the larger N, the closer to 36× (measured points on the predicted curve).
- **Contrast with EP3:** serial 0.15 s (context) → 34×; serial 5.6 s (a 5 GB image) → 19×.
- **Fair benchmarks:** the unit is the SM, not "a GPU vs one CPU core" (the 100× myth, Lee et al. 2010).
- **⟨A30:⟩** strong scaling over 56 SMs ⟨speedup, p⟩; the laptop vs the A30 per SM per GHz ⟨⟩; FP64 vs FP32 on each ⟨the flip⟩; the 1k run's serial part on IAG's disk ⟨⟩.
- **Checkpoint (predict first):** "With a 2k image, does the whole run scale better or worse?"

### 10. Result and closing (6 min)
- **Slides:** the SNR ladder (spots appear between 10⁹ and 10¹⁰ photons; SNR ∝ √N); ⟨A30: the 2k image⟩.
- **Three takeaways:** (1) GPUs win when the work is many independent pieces; (2) correctness first: the same bits on CPU and GPU; (3) Amdahl decides: find the serial part (the context, the image), measure the whole run, and name the baseline of every speedup.

## The arc (first draft, 2026-11-20; replaced by the sequence above)

The broad idea: **history → foundations → programming model → Buffon → the star**, with the tools (Nsight) as a thread through all of it. The order and weights will change.

| # | Block | Main idea | Board | Slides / demo |
|---|---|---|---|---|
| 0 | **Opening question** | "π with 14 decimals: 2.7·10¹¹ years on this GPU. Why so long, and why is the GPU still 100× faster than the CPU?" | the question | one number |
| 1 | **History** | From graphics cards to general-purpose computing. GPGPU through shaders, then CUDA (2007), FP64 and caches, then tensor cores, then GPUs in today's supercomputers | a timeline | photos of the cards, a Top500 chart |
| 2 | **Foundations** | Latency vs throughput: the CPU spends its transistors on caches and control, the GPU on many simple ALUs. Why that suits Monte Carlo | the CPU vs GPU chip sketch | - |
| 3 | **The programming model** (Programming Guide) | host/device, kernels, grid → blocks → threads → warps → SMs, SIMT | the thread hierarchy, drawn | a minimal kernel |
| 4 | **Memory** | two address spaces, `cudaMalloc`/`cudaMemcpy`, registers / shared / global / constant | the memory hierarchy, drawn | - |
| 5 | **Working together** | races, atomics, the work queue | the race, by hand | - |
| 6 | **Buffon** | everything above in one program: persistent threads, the queue, Philox streams per chunk, SM masking | - | code excerpts |
| 7 | **Measuring Buffon** | Amdahl with n = SMs, p depends on N, then Gustafson: grow the problem with the machine ([scaling_laws.md](../docs/scaling_laws.md)), FP64 vs FP32 | **the class predicts first**, then we show | plots from EP3, **live nsys / ncu** |
| 8 | **The star: new problems** | a 1.5 GB image, atomics on many addresses, constant memory, the RNG check between CPU and GPU | why caching does not help here | - |
| 9 | **The star: results** | whatever is ready: the video, the speedups, the profile | - | the video, plots |
| 10 | **Closing** | what the GPU is good for, what it is not, what we learned | the three takeaways | - |

## Ready material

- [`nsight.md`](nsight.md): how `nsys` and `ncu` work, and Buffon profiled (timeline in [`profiling/`](profiling/)). For block 7, live. Main finding: Buffon's serial part (~0.15 s) is CUDA context creation.
- [`../docs/precision.md`](../docs/precision.md) and [`../docs/gpus.md`](../docs/gpus.md): FP32 vs FP64, and the laptop vs the IAG A30 (the precision trade-off flips between them). For blocks 7 and 9.

## Teaching techniques

- **Predict, then measure.** Before each result, the class guesses (the speedup on 36 SMs? FP32 vs FP64?). Then the measurement. It keeps the room thinking, and our data is real.
- **Build at the board.** Draw the thread hierarchy and the memory spaces step by step rather than showing a finished diagram.
- **One idea per slide.** Most slides: one plot or one code excerpt, with a title that states the conclusion.
- **Live tools.** A short `nsys` timeline and an `ncu` Speed Of Light section on Buffon, run in front of the class (with screenshots as a backup).
- **Checkpoints.** A question to the room at the end of each block.

## Open questions

- [x] Date and length: **Tuesday 2026-11-24, 2 hours**, the CUDA class together with the projects (EP3 and EP4). EP4 is delivered with it.
- [ ] Is it the team class or the individual one? Who else is in the CUDA team, and how is the time split?
- [x] Language: the class and slides in Portuguese; all docs in English.
- [ ] Projector, board, can we run live demos on the laptop?
- [ ] How much of the star will be ready by then?

## Slides

- Tool: to decide (a PowerPoint file, or a deck made here that exports to .pptx/PDF).
- First step: settle the arc, then draft slide by slide from this plan, translating to Portuguese at that point.

## Log

- 2026-11-20: plan created. Arc from the first idea: history, foundations, Programming Guide, Buffon concepts, Buffon, star concepts, star.
- 2026-11-20: class language set: Portuguese.
- 2026-11-21: Buffon profiled with `nsys`; `ncu` waits for the GPU performance counters to be enabled on Windows.
- 2026-11-21: all docs kept in English; only the class and slides will be translated.
- 2026-11-23: the professor's syllabus transcribed; the sequence drafted around it (foundations, EP3 as delivered, the harder concepts, the star); presenting alone. The A30 enters as a prediction, measured at IAG on Monday.
- 2026-11-23: the slides move to HTML (reveal.js) with the phone as the remote; then rebuilt: 77 slides in 11 parts, ~25 diagrams, a new part on random numbers and their validation, and the thread history → foundations → projects. Every data point is produced by the orchestrator ([class_points.md](class_points.md)); the two gaps found (the phases of a run, instructions per register cap) were added to `measure`.
- 2026-11-21: date set: Tuesday 2026-11-24, 2 hours, CUDA + EP3 + EP4. The build plan up to it: [roadmap.md §0](../docs/roadmap.md#0-today-all-of-ep4-saturday-2026-11-21-to-review). EP4 is built on Saturday, the class prepared on Sunday, from [ep4_concepts.md](ep4_concepts.md).

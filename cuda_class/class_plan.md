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

## The arc (first draft)

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
| 7 | **Measuring Buffon** | Amdahl with n = SMs, p depends on N, FP64 vs FP32 | **the class predicts first**, then we show | plots from EP3, **live nsys / ncu** |
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

- [ ] Date and length of the class (the course plan gives team presentations about 1 h).
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

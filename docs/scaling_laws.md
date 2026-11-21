# Scaling laws: Amdahl and Gustafson

Two laws, two questions:

| | **Amdahl** (1967) | **Gustafson** (1988) |
|---|---|---|
| Question | How much faster does a **fixed problem** run on n processors? | How much **bigger a problem** can n processors solve in the same time? |
| Scaling | strong scaling | weak scaling |
| Speedup | S(n) = 1 / (s + p/n), at most 1/s | S(n) = s′ + p′·n = n − s′(n − 1), grows without limit |
| Message | the serial part caps the speedup | the serial part only costs a fixed slice; grow the problem with the machine |

They do not contradict each other: they measure different things.

## 1. Amdahl: a fixed problem

Run the program on one processor and split its time T(1) into a serial fraction s and a parallelizable fraction p (s + p = 1). On n processors only the parallel part shrinks:

```
T(n) = T(1) · (s + p/n)
S(n) = T(1) / T(n) = 1 / (s + p/n)  →  1/s   as n → ∞
```

With s = 10% the speedup never passes 10, however many processors. That is the course's starting point ("04 - Lei de Amdahl.pdf"), and EP3 measured it over SMs: p = 0.99864 at N = 3.2·10¹⁰ needles, so the ceiling is 1/(1 − p) ≈ 740.

**What Amdahl assumes:** the problem stays the same size as processors are added.

## 2. Gustafson: a problem that grows with the machine

John Gustafson, *"Reevaluating Amdahl's Law"*, Communications of the ACM 31(5), 1988, starts from the other end:

- **People do not buy a bigger machine to run yesterday's problem faster; they run a bigger problem in the same time.**
- So measure the fractions **on the parallel run**: on n processors the run takes time 1, split into a serial part s′ and a parallel part p′ (s′ + p′ = 1).
- On one processor, the same work would take s′ + n·p′: the serial part once, and the parallel work done by n processors, now done by one.

```
scaled speedup  S(n) = (s′ + n·p′) / (s′ + p′) = s′ + n·p′ = n − s′·(n − 1)
```

- **The speedup grows linearly with n,** with slope p′ = 1 − s′. With s′ = 10%, 100 processors give a scaled speedup of 90.1, not Amdahl's ceiling of 10.
- **s′ is not Amdahl's s.** s′ is the serial share *of the big parallel run*. As the problem grows, the parallel work grows and the serial share shrinks. That is exactly what EP3 saw from the other side: Amdahl's p went from ~0 at N = 10⁴ to 0.9986 at 3·10¹⁰.

## 3. Why Gustafson matters for us

1. **Our serial part does not grow with N.**
   - In Buffon it is CUDA context creation (~0.15 s, [measured with `nsys`](../cuda_class/nsight.md#22-where-is-the-serial-part-of-amdahls-law)).
   - Grow N and its share vanishes: the Gustafson situation, where adding SMs keeps paying off as long as the problem grows with them.
2. **Monte Carlo is the textbook weak-scaling problem.** More photons means less noise, so a bigger machine is naturally used for a *better* image, not a faster identical one.
3. **The star adds a twist: its serial part grows with the image.**
   - Zeroing the image, copying it back and writing it to disk are all proportional to the image size.
   - Gustafson's law assumes a serial part that stays fixed while the problem grows. For the star it holds when we grow **N** (more photons, same image) but not when we grow the **image** (512 → 1k → 2k).
   - Measuring that difference is a result in itself.
4. **Across machines, Gustafson's question is the natural one:** "what can the laptop do in one hour, and what can the A30 do in the same hour?" (1k with 4 observers vs 2k with 4 observers, for example). The planner's "largest image I can fill in a time budget" is Gustafson's question turned into a tool.

## 4. A check with EP3's numbers

EP3, all 36 SMs, N = 3.2·10¹⁰: wall time 5.115 s, serial part 0.175 s.

- **Amdahl, measured:** speedup 34.28 vs one SM.
- **Gustafson:** s′ = 0.175 / 5.115 = 0.034, so S(36) = 36 − 0.034 × 35 = **34.8**.

They are close here because N is so large that the serial share is already tiny either way. At small N they diverge: the fixed problem stops scaling (Amdahl), while a problem that grows with the SMs keeps scaling (Gustafson).

## 5. Experiments for EP4

| Experiment | Kept fixed | Grows | Law | Ideal result |
|---|---|---|---|---|
| **Strong scaling over SMs** | N, image | SMs: 1 → 36 (laptop), 1 → 56 (A30) | Amdahl | time ∝ s + p/n; fit p |
| **Weak scaling over SMs** | photons per SM, image | SMs, and N ∝ SMs | Gustafson | **constant time**; efficiency T(1 SM, N₁) / T(n SMs, n·N₁) |
| **Weak scaling over the image** | SNR (photons per pixel), all SMs | the image: 512 → 1k → 2k (4× pixels and 4× photons per step) | Gustafson, with a serial part that grows | time ×4 per step from the photons, plus serial costs that also grow ×4 |
| **Across machines** | a time budget | the problem each machine fills | Gustafson | the largest image each machine fills in the budget |

- **Weak-scaling efficiency:** E(n) = T(1, W) / T(n, n·W), where W is the work per processor. 1 is perfect; less than 1 measures the overhead that grows with n (more SMs competing for the image's atomics, the memory, the queue).
- **For the TCD:** plot strong and weak scaling side by side, and fit Amdahl's p and Gustafson's s′ from the same runs.
- **For the class:** the strong-scaling curve flattens, the weak-scaling line stays flat, and the question "which one is the right measure?" depends on what you want from a bigger machine.
- **How they are run and analysed:** as sweeps (`sweep`, `remote sweep`) and the `scaling` command, which fits Amdahl's p and Gustafson's s′ and checks them against the serial part the engine measures directly ([driver.md §6–7](driver.md#6-sweeps-the-runs-of-a-scaling-study)).

## References

- G. M. Amdahl, *"Validity of the single processor approach to achieving large scale computing capabilities"*, AFIPS Spring Joint Computer Conference, 1967.
- J. L. Gustafson, *"Reevaluating Amdahl's Law"*, Communications of the ACM 31(5), 532–533, 1988.
- Course slides: "04 - Lei de Amdahl.pdf".

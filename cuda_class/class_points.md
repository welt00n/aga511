# The class points, and where each comes from

Every number, figure or claim in the class, with the command that produces it. **The rule (2026-11-23): every point shown in the class is obtained by the orchestrator** (`EP4/main.py`: `test`, `make`, `measure`, `compare`), so that it can be re-made on any machine (the laptop, the A30) and checked. EP3's points come from EP3's own pipeline, as delivered (`EP3/buffon.py`). History and the CUDA facts come from their sources, which are cited.

Legend: ✓ the orchestrator produces it · **gap** the class uses it but only `research/` produces it today · — not data (history, definitions).

Paths: `M/` = a `measure` run folder (`runs/measure/<stamp>-<host>-all/`), `T/` = a `test` run, `S/` = a `make` run, `C/` = a `compare` run, `E3/` = `EP3/runs/<stamp>-gpu-fp64/`.

## The thread: history → foundations → projects

Each concept enters with the problem that created it, is explained in the foundations, and is then used, and measured, in a project. The class follows these threads; each is a bridge from one part to the next.

| Concept | Born (history) | Explained (foundations) | Used and measured (projects) |
|---|---|---|---|
| data parallelism: one computation per element | 1999, pixels per frame | the kernel, thread → block → grid | Buffon: a needle per iteration; the star: a photon per iteration |
| throughput, not latency | 1999, many simple units | CPU vs GPU; latency hiding by warps | FP64 bound by its units, not by latency (registers sweep) |
| the kernel | 2001, a program per pixel | `__global__`, the launch | `throw_needles`, `trace_photons` |
| a real programming model | 2004, GPGPU through textures | host/device, two memories, the flow | the phases of a run: setup, photons, copy, save |
| the SM | 2006, G80 | the SM's units; blocks go to SMs | **the unit of every speedup**: 1 SM vs n SMs (the SM mask) |
| FP64 and its cost | 2008, Tesla / Fermi | FP32 vs FP64 bits; separate units, 1:64 vs 1:2 | 16.2× FP32/FP64 here; the photons moved by FP32; the A30 flips it |
| the memory hierarchy | 2010, Fermi caches, ECC | registers, shared, constant, global | the star in constant memory, the image in global, the photon in registers |
| cooperation and its price | — | reduction, `__syncthreads`, races, atomics | Buffon: one atomic per thread; the star: an atomic per photon on the image |
| random numbers in parallel | (Monte Carlo needs them: EP1, EP3) | stateful vs counter-based; Philox | a stream per chunk: the same image on CPU and GPU, any split, any SMs |
| specialized units | 2017, tensor cores | the SM has FP32, INT32, FP64, tensor | the instruction mix: our kernel is integer + FP32 + FP64 work |
| energy as the limit | 2022, exascale | the clock is not fixed | the laptop's power cap: FP32 on all SMs at 2730 MHz, hence cycles |
| Amdahl | (EP1, EP3) | the serial part sets the ceiling | EP3: the context, 34×; EP4: the image, 19×, toward 36× at large N |

## Part by part

### 1. History (—)
Facts from the sources, no data: Cray-1 (1976), Connection Machine CM-1 (1985, 65 536 one-bit processors), GeForce 256 (1999), GeForce 3 (2001, programmable shaders), BrookGPU (2004), G80 (2006) and CUDA (2007), FP64 in GT200 (2008), Fermi (2010: ECC, L1/L2), AlexNet (2012, two GTX 580), Volta (2017, tensor cores), Frontier (2022, the first exascale machine). Lee et al., ISCA 2010, for the "100×" myth.

### 2. Architecture
| Point | Source | Status |
|---|---|---|
| this GPU: name, SMs, compute capability, memory | every GPU product prints it (`environment.show_launch`); `M/machine.json` | ✓ |
| per SM: 65 536 registers, 48 warps (1536 threads), 24 blocks, 102 400 B shared | `measure` occupancy part: `M/parts/occupancy.json` → `device` | ✓ |
| the A30's limits (56 SMs, 64 warps, 32 blocks) | `KNOWN_TARGETS["80"]` (Ampere tuning guide), printed by `measure --targets 80` | ✓ |
| units per SM (128 FP32/INT32 and 2 FP64 here; 64 + 64 + 32 on the A30) | the datasheets / whitepapers | — |
| FP32 vs FP64: 24 vs 53 bits of mantissa | IEEE 754 | — |

### 3. Terminology and memory
| Point | Source | Status |
|---|---|---|
| our launch: FP64 80 registers → 25 warps allowed, 24 launched (blocks fit only whole), 768 threads × 1 block per SM, occupancy 0.50, 27 648 threads; FP32 67 → 28 warps, 0.58 | the launch line of every GPU product; `M/parts/kernel.json` → `resources` | ✓ |
| the star's description in constant memory (bytes) | `M/parts/kernel.json` → `resources.*.constant_bytes` | ✓ |
| constant vs global for the star (speed) | `measure` kernel part, variant `scene_in_global` | ✓ |

### 4. The execution flow
| Point | Source | Status |
|---|---|---|
| EP3: the serial part 0.15 s is creating the CUDA context | `E3/summary.txt` (the time model); `nsys` ([nsight.md](nsight.md)) | ✓ (EP3) |
| EP4 at 1k, FP64, 1 SM and 36 SMs: setup, zeroing, photons, copy back, save, each in seconds and as a share of the whole run | `M/parts/scaling.json` → `strong-fp64.phases`; `M/figures/phases_fp64.png`; the report's section 5.1 | ✓ (since 2026-11-23) |

### 5. The programming model
| Point | Source | Status |
|---|---|---|
| SIMT, divergence, the reduction, races, atomics | the CUDA Programming Guide | — |
| our kernel has no `__syncthreads`; it cooperates only through atomics | the code (`gpu.cu`) | — |

### 6. Random numbers in parallel
| Point | Source | Status |
|---|---|---|
| our Philox4x32-10 (CPU) = Random123's published answers (3 known-answer tests) | `main.py test` → "rng cpu"; the engine's `--test-rng` | ✓ |
| cuRAND's Philox (GPU) = ours, bit for bit: chunks 0, 1, 1 000 003, 4 294 967 301, the bits and the uniforms (`memcmp`) | `main.py test` → "rng gpu" | ✓ |
| any split of the chunks, any number of passes: the same image, bit for bit | `main.py test` → "split", "passes" | ✓ |
| the same image on 1, 2, 9 and 36 SMs | `measure` validation → `digests`; `compare` → image digests | ✓ |
| the GPU's image = the CPU's, bit for bit (FP64, 10⁸) | `main.py test` → "gpu = cpu"; `M/parts/validation.json` | ✓ |
| independent photons: χ²/dof 0.9904 over 51 899 blocks (−1.6σ), the figure | `M/parts/validation.json`; `M/validation/independent-gpu64/comparison.png` | ✓ |
| FP32 on the GPU vs the CPU, the same photons: 2.98·10⁻⁴ of the photons move (2484 pixels at 1k, 10⁸) | `M/parts/validation.json` → the `gpu` check at FP32 | ✓ |
| why FP32 differs: the fused multiply-adds (~3/4) and CUDA's FP32 math library | the `no_fma` kernel variant: `M/parts/kernel.json` → `variants` (speed); the photons it moves: research (walkthrough piece 18) | the reason is research; say it as such |
| EP3: the error falls as N^(−0.499±0.009), theory 2.37/√N | `E3/error.png`, `E3/summary.txt` | ✓ (EP3) |
| Philox passes TestU01's BigCrush | Salmon et al., SC 2011 (we don't run BigCrush) | — |

### 7. EP3: Buffon (as delivered)
| Point | Source | Status |
|---|---|---|
| 34.3× on 36 SMs, p = 0.9986, N = 3.2·10¹⁰ (FP64); p grows with N | `E3/amdahl.png`, `E3/summary.txt` | ✓ (EP3) |
| the code: Philox per chunk, persistent threads, a queue, one atomic per thread | `EP3/buffon.cu` | — |

### 8. Deeper: warps, registers, occupancy
| Point | Source | Status |
|---|---|---|
| registers → warps that fit → occupancy (the arithmetic) | `M/parts/occupancy.json` → `this_gpu`, `rows`; the launch line | ✓ |
| the register sweep: more occupancy is not faster (FP64 flat to 0.67, then spills cost up to 10%; FP32 fastest uncapped) | `measure` occupancy part, `M/figures/registers_and_occupancy.png` | ✓ |
| **why**: capping adds instructions before any spill (+4.9% at 55 registers in FP32, 0 B spilled), then spills (local stores and loads) | `M/parts/occupancy.json` → `rows.*.instructions_vs_uncapped`, `local_stores`, `local_loads`; the report's section 4 | ✓ (since 2026-11-23; first found by `research/register_cap_instructions.py`) |
| FP64 is 16.2× slower than FP32 here, not 64× | `M/parts/validation.json` → precision `speed_ratio`; `M/parts/speed.json` | ✓ |
| the FP64 kernel's `NOP`s are waits for the FP64 units | the static mix in `M/parts/kernel.json` → `instruction_mix` (counts); the "why" in `research/nop_runs.py` | ✓ counts, the reading is research |
| registers = the peak of live values (74 live in FP64, inside sin/cos) | `research/register_liveness.py` | **gap** (optional for the class) |
| the A30 needs 96 / 76 registers: occupancy ≤ 0.33 / 0.39 | `measure --targets 80` → `M/parts/occupancy.json` → `targets` | ✓ |
| why 96 on sm_80: the compiler's front end changes at compute_100 | `research/front_end_registers.py` | research only (a remark, not a slide) |
| the A30 measured: occupancy, sweep | `measure` on the A30 | ⟨A30⟩ |

### 9. EP4: the star, the code
| Point | Source | Status |
|---|---|---|
| the star's videos, 7 observers | `make` → `S/observers_around.mp4`, `S/observers_grid.mp4` | ✓ |
| 3.9·10¹² photons for SNR 10 at 1k, 6.3 min on 36 SMs | `make --snr 10 --plan` (the prediction) and the run's record | ✓ |
| SNR measured = predicted | `make` → the SNR check in `S/` | ✓ |
| the photon's steps and their shares of the time (sections) | `M/parts/kernel.json` → `sections` | ✓ |
| the blur of the bins (11 px at the equator, up to 105 px in the polar bin) | `measure` costs part → `M/parts/costs.json` | ✓ |

### 10. EP4: Amdahl and the comparison
| Point | Source | Status |
|---|---|---|
| photons 216.9 s → 6.0 s (36.3×), whole run 223.5 → 11.6 s (19.3×), p = 0.974, ceiling 38 | `M/parts/scaling.json` → `strong-fp64`; `M/figures/amdahl_ceiling_fp64.png` | ✓ |
| in cycles 36.1× (FP64), 36.3× (FP32); the clock during the photons 2878 / 2730 MHz | `M/parts/scaling.json` → `speedup_cycles`, `sm_clock_during_photons_mhz` | ✓ (since 2026-11-23) |
| the power cap holds FP32 on all SMs | `M/parts/speed.json` → `power_cap_share`; per point `GPU_MONITOR.POWER_CAP_SHARE` | ✓ |
| large N: toward the ideal, measured points on the model (FP32 at 10¹¹: 21.4× vs 22.2×) | `M/parts/costs.json` → `large_n`; `M/figures/costs_large_n.png` | ✓ |
| EP3 vs EP4: 0.15 s / 34× vs 5.6 s / 19× | `E3/summary.txt`; `M/parts/scaling.json` | ✓ |
| the laptop vs the A30: speeds, per SM per GHz, scaling, occupancy, image digests | `compare M_laptop M_a30` → `C/` | ⟨A30⟩ |
| the unit is the SM, not a CPU core | `measure` speed part (one SM vs all) | ✓ |

### 11. Closing
| Point | Source | Status |
|---|---|---|
| the SNR ladder: the spots appear between 10⁹ and 10¹⁰ | `make --ladder 1e8:1e11` → `S/snr_ladder.mp4`, its SNR table | ✓ |
| the 2k image on the A30 | `make --size 2048 ...` on the A30 | ⟨A30⟩ |

## The gaps, and how they were closed (2026-11-23)

The inventory found three points that the class uses but that only `research/` produced:

1. **The phases of a run** (setup, zeroing, photons, copy back, save) for 1 SM and all SMs: now in `measure`'s scaling part (`strong-fp*.phases`, `figures/phases_fp*.png`, report section 5.1). Analysis only, from the points already measured.
2. **Instructions per register cap**, with the local stores and loads, next to the sweep's cycles: now in the occupancy part (`rows.*.instructions_vs_uncapped`, report section 4). Compile-only: it reads the cap builds the sweep already makes.
3. **The FP32 GPU-vs-CPU difference**: it was already there, as the `gpu` check at FP32 in `measure` validation.

The 1k laptop measurement was re-analysed with `main.py measure --continue` (0 of 178 points to run), so its figures and report carry all of it; the A30's `measure` produces it directly.

What stays research and is said as such in the class: the reading of the `NOP`s (waits for the FP64 units), the live-register peak inside sin/cos, and the compiler's front end at compute_100.

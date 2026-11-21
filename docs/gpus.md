# The GPUs: the laptop and the IAG A30

EP4 will run on two GPUs: the RTX 5070 Laptop (development) and an NVIDIA **A30** at IAG (reached by SSH from inside the IAG network). They are built for different jobs, and that is material for the TCD and the class.

## Side by side

| | RTX 5070 Laptop | A30 |
|---|---|---|
| Architecture | Blackwell, compute capability 12.0 | Ampere (GA100, the A100's chip), compute capability 8.0 |
| SMs | 36 **[measured]** | 56 |
| Max threads per SM | 1 536 **[measured]** | 2 048 |
| Threads resident at once | ~55 000 | ~115 000 |
| Memory | 8 GB GDDR7 **[measured]** | 24 GB HBM2, ~930 GB/s |
| FP32 peak | ~20 TFLOPS (estimate from cores × clock) | ~10 TFLOPS (datasheet) |
| FP64 peak | ~0.3 TFLOPS | ~5 TFLOPS (datasheet) |
| FP64 : FP32 | 1 : 64 **[measured: CUDA attribute]** | 1 : 2 |
| MIG (split into slices) | no | yes, up to 4 |

Laptop values marked [measured] come from `EP3/buffon -i`. The A30 values are from NVIDIA's datasheet, to be confirmed on the machine.

## Predictions to test

1. **"Which GPU is faster?" depends on the precision.**
   - In FP64 the A30 should beat the laptop by ~10–15×: it has full double-precision hardware.
   - In FP32 the laptop may be as fast or faster: more FP32 throughput, a newer architecture, higher clocks.
   - The same code, on two GPUs, gives opposite answers to "should we use FP32?".
2. **Amdahl over SMs has 56 points instead of 36**, if the A30 is not split with MIG.
3. **The same star, bit for bit, on both GPUs.**
   - Same seed and chunks, Philox, integer counts: the images should be identical.
   - At most a few pixel-edge photons may differ if the compilers round differently.
   - It also proves the A30 run is correct.

## What the code must do to scale there unchanged

1. **Never assume the hardware.** SM count, block size, blocks per SM and threads per SM come from the occupancy API at run time, as EP3's `choose_launch` does. EP3's `cuda_cores_per_sm` already returns 64 for compute capability 8.0, which is right for the A30.
2. **Build on the machine that runs it, for its exact GPU, without PTX** **[decided 2026-11-21]**.
   - **Two kinds of GPU code** can go into a binary:
     - **SASS** (a `cubin`): machine code for one compute capability (cc). `sm_80` SASS runs on cc 8.x, not on 9.0 or 12.0.
     - **PTX**: NVIDIA's portable assembly. The driver compiles it to SASS at the first launch (JIT, cached in `~/.nv/ComputeCache`), so it runs on newer GPUs too.
   - **We build where we run**, so SASS for that GPU is enough: `-arch=native` on the laptop (measured: EP3's `buffon` contains `sm_120` cubins only, no PTX), `make ARCH=sm_80` on the A30 (CUDA 13.3 can build it: `nvcc --list-gpu-arch` includes `compute_80`).
   - **Why no PTX:**
     - A wrong build **fails at once** ("no kernel image is available"), which the preflight catches ([roadmap.md](roadmap.md#preflight-can-we-run-here-local-and-remote-asked-2026-11-21)), instead of quietly JIT-compiling.
     - **No JIT time hidden in the first launch:** it would land in the serial part of Amdahl's law, and differ between the first run and the next.
   - **Building on the target also matches the toolkit to the driver.** A binary from the laptop's CUDA 13.3 needs a driver that supports CUDA 13; if IAG's driver is older, it would not run there even with the right SASS. IAG's own `nvcc` avoids the question (preflight check 2).
   - **If IAG has no toolkit:** build on the laptop with `ARCH=sm_80`, and the driver there must support CUDA 13.
   - **Record what was built:** the cc, the toolkit and the driver versions go into the summary and the records, so two runs can be compared knowing what made them.
3. **Fill the GPU.** ~115 000 threads with chunks of 1 024 photons means N ≳ 10⁸ to give each thread one chunk.
4. **One command per session, with a time budget.** Access is limited, so the driver runs every experiment (SM sweep, N sweep, FP32/FP64, resolutions) within `--budget`, as EP3's driver does, and saves everything.

## Questions for IAG

- [ ] Is the A30 whole, or split with MIG? (`nvidia-smi` shows it.)
- [ ] SSH to a workstation, or jobs on a cluster with a scheduler? Any time limit per job?
- [ ] CUDA toolkit version and driver.
- [x] ~~Python with numpy/matplotlib?~~ Not needed: the server only runs the engine; analysis is local ([driver.md](driver.md#remote-remote-command-host)). Needed instead: `bash`, `make`, `nvcc`, `rsync` or `git`, `setsid`/`nohup`.
- [ ] Can we `git clone` the repo from there?
- [ ] Are `nsys` and `ncu` installed, and are the GPU performance counters allowed?

## How big can the image be?

### There is no "Max image" in the code

- **The "Max" setup** (1024², 360 frames, 180 μ bins, 4 observers, 6 GB) was only a row in a planning table.
- **The engine's option limits are sanity checks, not memory limits:** `nx`, `ny`, `nphi` ≤ 65 536, `nmu` ≤ 4 096, observers ≤ nmu.
- **The real limit is whether the allocation succeeds.** On the CPU, `allocate_image` stops with "Could not allocate the image (X GB)". On the GPU it will be `cudaMalloc` failing.

### Three limits, in the order they bite

**1. Memory: the image must fit.**

```
image bytes = 4 · nx · ny · nphi · observers
```

| Where it must fit | Laptop | A30 |
|---|---|---|
| GPU memory | 8 GB, **7.8 GB free** at idle **[measured, `nvidia-smi`]** | 24 GB |
| Budget after the CUDA context and headroom | ~7 GB | ~22 GB |
| RAM, for the copy back and for Python | 31 GB in WSL **[measured]** | to check at IAG |
| Disk, for the `.bin` | 937 GB free **[measured]** | to check at IAG |

- The GPU is the tightest.
- RAM must hold the copy back, and Python more: `np.fromfile` holds the whole image, and the pictures convert one observer at a time to float64 (8 bytes per pixel).

**2. Photons: the image must be filled.**

```
photons for 100 counts per pixel ≈ 100 · nmu · nphi · (π/4) · nx · ny
```

Doubling the side of the image needs 4× the photons for the same noise. Memory decides what *fits*; time decides what you can *fill*.

**3. Time: photons / speed.** The GPU speed is still to be measured; the table uses 10⁹ photons/s as a round guess.

### Examples (360 frames, 180 μ bins, 4 observers)

| Image | Memory | Fits | Photons for 100 counts/pixel | Time at 10⁹ photons/s |
|---|---|---|---|---|
| 512² (HD) | 1.5 GB | laptop, A30 | 1.3·10¹² | ~22 min |
| 1024² | 6.0 GB | laptop (tight), A30 | 5.3·10¹² | ~1.5 h |
| **~1100²** | 7.0 GB | **largest on the laptop** | 6.2·10¹² | ~1.7 h |
| 1448² | 12.1 GB | A30 | 1.1·10¹³ | ~3 h |
| **~1950²** | 22 GB | **largest on the A30** | 1.9·10¹³ | ~5.3 h |
| 2048² | 24.2 GB | neither, in one piece | 2.1·10¹³ | ~6 h |

Largest square side with `nphi` frames and K observers: side ≈ √(budget / (4 · nphi · K)).

### Ways to trade

- **Fewer frames** (`nphi`): memory and photons both shrink proportionally (each frame bin collects more photons).
- **Fewer observers:** memory shrinks, photons do not. Each observer's bin collects the same share of photons however many observers there are.
- **Fewer μ bins** (`nmu`): no memory saved, but fewer photons needed, because more photons fall in each observer's bin. The cost is more blur in latitude (each bin is wider).
- **Splitting by observers:**
  - run the *same chunks* several times, each run recording a subset of observers into a smaller image;
  - the photons are identical (same seed and chunks), so the pieces are exact;
  - each pass re-simulates all photons, so the time multiplies by the number of passes.
- **Splitting by frames** (ranges of `iphi`) works the same way.

### For the GPU engine **[to do]**

- Query the free memory (`cudaMemGetInfo`) before allocating, and report it in the result line (`IMAGE_BYTES`, `GPU_FREE_BYTES`). A too-large image should fail with a clear message.
- Let the driver plan: given a time budget and the measured speed, pick the largest image it can fill, or split by observers.

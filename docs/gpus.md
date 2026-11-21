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
2. **A portable build.** `-arch=native` builds for the GPU present. Add an `ARCH` option (`make ARCH=sm_80`), and avoid features newer than the CUDA version installed at IAG (Ampere needs CUDA 11 or newer).
3. **Fill the GPU.** ~115 000 threads with chunks of 1 024 photons means N ≳ 10⁸ to give each thread one chunk.
4. **One command per session, with a time budget.** Access is limited, so the driver runs every experiment (SM sweep, N sweep, FP32/FP64, resolutions) within `--budget`, as EP3's driver does, and saves everything.

## Questions for IAG

- [ ] Is the A30 whole, or split with MIG? (`nvidia-smi` shows it.)
- [ ] SSH to a workstation, or jobs on a cluster with a scheduler? Any time limit per job?
- [ ] CUDA toolkit version and driver.
- [ ] Python with numpy/matplotlib, or can we create a conda environment?
- [ ] Can we `git clone` the repo from there?
- [ ] Are `nsys` and `ncu` installed, and are the GPU performance counters allowed?

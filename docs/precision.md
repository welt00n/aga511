# Floating-point precision: FP64, FP32, FP16 and random bits

Do we need FP64 for the star? Short answer: **probably not**. This file explains why, and what EP4 does to measure it instead of assuming it.

## 1. The formats

IEEE 754 stores a number as sign × mantissa × 2^exponent. A `double` has 64 bits: 1 sign, 11 exponent and 52 stored mantissa bits, plus an implicit leading 1, so **53 bits of precision**.

| Format | Precision bits | Unit roundoff (relative error of one operation) | Decimal digits | Largest value |
|---|---|---|---|---|
| FP64 `double` | 53 | 2⁻⁵³ ≈ 1.1·10⁻¹⁶ | ~16 | ~1.8·10³⁰⁸ |
| FP32 `float` | 24 | 2⁻²⁴ ≈ 6.0·10⁻⁸ | ~7 | ~3.4·10³⁸ |
| FP16 `half` | 11 | 2⁻¹¹ ≈ 4.9·10⁻⁴ | ~3 | 65 504 |

## 2. Random bits are not the same as the type

A uniform random number is built from random **bits**. The number of random bits sets how fine its grid is, whatever type stores it.

- **53 random bits:** ξ = k·2⁻⁵³, about 9·10¹⁵ equally spaced values in [0,1). That is the full precision of a double.
- **32 random bits:** ξ = k·2⁻³² + offset, about 4.3·10⁹ values spaced 2.3·10⁻¹⁰.

What cuRAND does with Philox (read from `curand_uniform.h`, `curand_globals.h`, CUDA 13.3):

| Function | Bits used | Formula | Interval |
|---|---|---|---|
| `curand_uniform` (float) | 32 (one draw) | `x * 2^-32 + 2^-33` | (0, 1] |
| `curand_uniform_double` | **32 (one draw)** | `x * 2^-32 + 2^-32` | (0, 1] |
| `curand_uniform2_double` | 53 (two draws per double) | `(x ^ (y << 21)) * 2^-53 + 2^-54` | (0, 1) |

So `curand_uniform_double` returns a double that only carries 32 bits of randomness.

### Does the star need more than 32 random bits?

Compare the grid with the **smallest physical scale** in the problem:

- a pixel: 1/512 ≈ 2·10⁻³ (HD);
- an observer bin: 2/nmu ≈ 10⁻², and 2π/nphi;
- a spot radius: 0.05 rad.

All are millions of times coarser than 2.3·10⁻¹⁰. Each photon also combines 5–6 independent draws, so even with 10¹² photons (each 32-bit value drawn ~250 times) the positions are only quantized at ~10⁻¹⁰, far below a pixel. **32 bits are enough.** 53 bits would only matter for quantities that are very sensitive to tiny ξ, such as `log(ξ)` in exponential sampling, which the star does not use.

**EP4 decision:** use cuRAND's standard 32-bit uniforms (`curand_uniform_double` for FP64, `curand_uniform` for FP32), and the same conversion in the CPU copy.

## 3. Why Monte Carlo is forgiving

Each photon is a short, independent calculation (~50 operations), and the tally is an **integer** count. Rounding errors do not build up from photon to photon, as they would in a long time integration, and there is no need for Kahan summation. What matters is only the error inside one photon, and whether it introduces a **bias**.

## 4. Where precision actually bites

1. **Bias at pixel edges.**
   - An FP32 relative error of ~6·10⁻⁸ only moves photons that land within ~10⁻⁷ of a pixel edge, about 10⁻⁵ of them.
   - The effect on the image is ~10⁻⁵. Poisson noise at 100 counts is 10%, and detecting a 10⁻⁵ bias would need ~10¹⁰ counts per pixel.
   - **Invisible.**
2. **Cancellation near the poles.**
   - `sqrt(1 - z*z)` subtracts two nearly equal numbers when z → 1. In FP32, the radius near the pole is resolved to ~3·10⁻⁴, a tiny artifact still below a pixel.
   - Fix: `sqrt((1 - z) * (1 + z))`, which keeps the small difference exact.
   - The same applies to the professor's `raiz = sqrt(1 - w*w)` for observers near the poles, and to `sin θ = sqrt(1 - μ²)`.
3. **Small spots.**
   - `cos_angle > cos(radius)` compares numbers close to 1. In FP32, the spot edge is resolved to ~6·10⁻⁸ / sin r. That is about 10⁻⁶ rad for r = 0.05 (fine), but **about 6% of the radius for r = 10⁻³**.
   - The precision needed depends on the smallest physical scale.
   - Fix: compare the squared chord instead, |p − s|² < 2(1 − cos r) = 4 sin²(r/2). The difference p − s of two nearby unit vectors is computed accurately.
4. **Why EP3's FP32 hit a floor.**
   - Buffon estimates π from the *ratio* of crossings, and the FP32 comparison `x <= L/2·sinf(θ)` has a systematic bias of ~10⁻⁷.
   - Once the statistical error falls below the bias (around 7 digits), more needles do not help.
   - That was a bias, not accumulated rounding. The star only needs SNR ~10–30 per pixel, so it never gets near such a floor.
5. **FP16 cannot work here.**
   - Its spacing near 1 is ~10⁻³, which cannot even place a photon inside a 512-pixel image (pixel = 2·10⁻³).
   - It also cannot hold counts above 65 504.

## 5. What precision costs on this GPU

- **The FP64 rate.** The RTX 5070 Laptop runs FP64 at 1/64 of the FP32 rate. **[measured: the CUDA attribute in EP3]** EP3 measured a 16× gap between FP64 and FP32 Buffon, not 64×, because Buffon also spends time on integer work (RNG, loop, atomics).
- **FP64 math functions.** The star uses more floating-point work (sin, cos, sqrt, cbrt, atan2), and an FP64 `sin` is a software sequence of many FP64 operations, so the gap may be closer to 64×. **[prediction, to measure]**
- **Registers.** A double takes two registers, which can lower occupancy.

## 6. What EP4 does

1. **A `real` type**, `double` by default, `float` when compiled with `-DSTAR_FP32` (`engine/real.cuh`). The physics is written once and works for both.
2. **FP64 on the CPU is the reference.**
3. **FP32-friendly formulas from the start:** `(1 - z)(1 + z)` under square roots, and the squared chord for spots.
4. **Experiment:** FP32 vs FP64 at the same seed and N, a per-pixel χ² (expected ≈ 1: no difference beyond noise), and the speed ratio. This is a TCD section: *the physical scales say FP32 is enough, and here is the measurement*.

## 7. First measurement (CPU, N = 10⁸, EP2's default star) **[measured]**

`python star.py -n 1e8 --compare-precision`:

- **Same random numbers, same photons.** Recorded photons: 2 652 017 in both. Photons in spots: 411 940 (FP64) vs 411 941 (FP32), one photon at a spot edge.
- **5·10⁻⁴ of the recorded photons land in a different pixel or bin** in FP32. That is larger than the rough 10⁻⁵ guessed in section 4. The rounding inside a photon (trig of angles up to 2π, several chained operations) is closer to 10⁻⁶ than to one unit roundoff, and every photon crosses four binnings (ix, iy, imu, iphi).
- **Still far below the noise.**
  - The relative Poisson noise per lit pixel (frames summed) is 0.22.
  - Two FP64 runs with different seeds (noise alone) give χ²/dof = 1.0007.
  - Rounding moves 5·10⁻⁴ of the photons: hundreds of times below the noise.
- **Speed on one CPU core:** FP32 is about 1.6× faster than FP64. The GPU gap is still to be measured (prediction: much larger, up to the 64× hardware ratio).

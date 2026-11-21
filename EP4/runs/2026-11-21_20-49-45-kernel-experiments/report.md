# Kernel experiments (the laptop GPU; the whole GPU, one pass, 4 observers, 200², 150 frames)

## FP64, N = 5e+08

| variant | registers | spills (B) | threads/block | occupancy | kernel time (median, min–max) | photons/s | vs baseline | SM clock (MHz) | power (W) | pixels ≠ baseline |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 80 | 0 | 768 | 0.500 | 0.7745 s (0.7744–0.7770) | 6.46e+08 | 1.000× | 2812 | 55.2 | 0 |
| scene_in_global | 88 | 0 | 640 | 0.417 | 0.7732 s (0.7714–0.7734) | 6.47e+08 | 1.002× | 2512 | 44.6 | 0 |
| registers_64 | 64 | 24 | 1024 | 0.667 | 0.7834 s (0.7799–0.7838) | 6.38e+08 | 0.989× | 2722 | 42.8 | 0 |
| registers_40 | 40 | 344 | 768 | 1.000 | 0.8503 s (0.8489–0.8507) | 5.88e+08 | 0.911× | 2584 | 59.1 | 0 |
| no_fma | 80 | 0 | 768 | 0.500 | 0.8353 s (0.8316–0.8368) | 5.99e+08 | 0.927× | 2756 | 55.2 | 0 |
| sections | 112 | 0 | 512 | 0.333 | 0.7698 s (0.7688–0.7723) | 6.5e+08 | 1.006× | 2486 | 46.8 | 0 |

**Where the cycles go (the `sections` variant):** surface 20.2%, direction 33.4%, spots 8.4%, bin 21.5%, pixel 1.4%, chunk 0.0%, record 15.0%

## FP32, N = 5e+09

| variant | registers | spills (B) | threads/block | occupancy | kernel time (median, min–max) | photons/s | vs baseline | SM clock (MHz) | power (W) | pixels ≠ baseline |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 67 | 0 | 896 | 0.583 | 0.4510 s (0.4492–0.4523) | 1.11e+10 | 1.000× | 2310 | 72.0 | 0 |
| scene_in_global | 68 | 0 | 896 | 0.583 | 0.4584 s (0.4583–0.4637) | 1.09e+10 | 0.984× | 2224 | 74.2 | 0 |
| registers_64 | 55 | 0 | 576 | 0.750 | 0.4675 s (0.4620–0.4704) | 1.07e+10 | 0.965× | 2531 | 72.9 | 0 |
| registers_40 | 40 | 80 | 768 | 1.000 | 0.4739 s (0.4729–0.4749) | 1.06e+10 | 0.952× | 2467 | 89.5 | 0 |
| no_fma | 67 | 0 | 896 | 0.583 | 0.4604 s (0.4570–0.4630) | 1.09e+10 | 0.980× | 2434 | 76.1 | 30689 |
| sections | 69 | 0 | 896 | 0.583 | 0.4894 s (0.4869–0.4901) | 1.02e+10 | 0.922× | 2310 | 33.9 | 0 |

**Where the cycles go (the `sections` variant):** surface 27.6%, direction 40.8%, spots 9.2%, bin 8.4%, pixel 0.9%, chunk 0.0%, record 13.2%

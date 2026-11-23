import common
from lab import engine

EQUATOR_SIDE = [1.5708, 1.4, 1.2, 1.0]
POLE_SIDE = [0.8, 0.6, 0.4, 0.0]
ORDERS = {"equator side first": EQUATOR_SIDE + POLE_SIDE, "pole side first": POLE_SIDE + EQUATOR_SIDE}
PHOTONS = (1e7, 1e8)
REPEATS = 3


def pass_kernels(observers, photons):
    config = engine.default_star_config(photons, observers=observers, observers_per_pass=4)
    output = common.engine_output(config, "gpu", 64)
    kernels = engine.lines_of_kind(output, "KERNEL")
    cycles = engine.lines_of_kind(output, "SMS")
    return [(kernel["KERNEL_TIME"], sms["CYCLES"]) for kernel, sms in zip(kernels, cycles)]


with common.recorded(__file__):
    print("Q3: is the second pass slower than the first? (2 passes of 4 observers, FP64, all SMs)")
    print("Each pass's kernel time and its cycles; the same cycles at a different time means the clock moved.\n")
    for photons in PHOTONS:
        for label, observers in ORDERS.items():
            for repeat in range(REPEATS):
                (time_1, cycles_1), (time_2, cycles_2) = pass_kernels(observers, photons)
                print(f"N={photons:.0e} {label:19s} pass 1 {time_1 * 1e3:8.2f} ms {cycles_1:.3e} cycles   "
                      f"pass 2 {time_2 * 1e3:8.2f} ms {cycles_2:.3e} cycles   pass 2 / pass 1 {time_2 / time_1:.3f}",
                      flush=True)
    print("\nFound on 2026-11-22: the 5x was not reproduced. At N=1e7 pass 2 / pass 1 ranged 0.63-1.60 with the cycles")
    print("in proportion (same clock, uneven work: 9 766 chunks for 27 648 resident threads); at 1e8 0.99-1.00.")

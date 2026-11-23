import common

with common.recorded(__file__):
    print("The SM's limits, from cudaGetDeviceProperties (what decides how many warps fit):\n")
    for key, value in common.sm_limits().items():
        print(f"  {key} = {value}")
    print("\nFound on 2026-11-22 (RTX 5070 Laptop, cc 12.0): 65 536 registers per SM, 1536 threads (48 warps),")
    print("24 blocks per SM. Registers are given in steps of 8 per thread (256 per warp).")

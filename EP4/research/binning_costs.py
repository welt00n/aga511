import math

import common
from lab import budget

CASES = [(1024, 300), (2048, 300), (2048, 600), (1024, 1000)]
INCLINATIONS = (0, 30, 60, 90)
SNR = 10


def describe(size, frames):
    nmu = budget.nmu_for(frames)
    print(f"== {size}², {frames} frames, nμ {nmu}: rotation blur {budget.rotation_blur(size, frames):.1f} px; "
          f"N for SNR {SNR} at the limb {budget.photons(size, frames, nmu, SNR):.2e}; "
          f"{budget.image_bytes(size, frames, 1) / 1e9:.2f} GB per observer")
    for degrees in INCLINATIONS:
        center, low, high = (math.degrees(angle) for angle in budget.inclination_bin(math.radians(degrees), nmu))
        print(f"   asked {degrees:3d}° → bin center {center:5.1f}°, spans {low:5.1f}–{high:5.1f}°: up to "
              f"{budget.tilt_blur(size, math.radians(degrees), nmu):5.1f} px of tilt blur")


with common.recorded(__file__):
    print("How much does the binning blur the image? A frame blends every photon of its cell (2π/frames of rotation ×")
    print("2/nμ of μ), each projected along its own direction (engine/physics/photon.cuh). R = size/2 pixels.\n")
    for size, frames in CASES:
        describe(size, frames)
    print("\nFound on 2026-11-22: at 1024², 300 frames, nμ 95 the blur is ~11 px along the rotation and at the equator,")
    print("but the polar bin spans 0–11.8° (up to 105 px). Equal-μ bins are equal in photons, not in angle. Keeping the")
    print("blur in pixels while doubling the size costs 16× the photons (N ∝ nμ · frames · size²).")

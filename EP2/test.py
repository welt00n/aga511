import numpy as np
import matplotlib.pyplot as plt

filename = "star_2026-09-25_11-30-59.bin"

nx = 256
ny = 256
nphi = 2
nmu = 2

data = np.fromfile(filename, dtype=np.uint32)

expected_size = nphi * nmu * ny * nx

if data.size != expected_size:
    raise ValueError(
        f"Expected {expected_size} values, got {data.size}"
    )

image = data.reshape(nphi, nmu, ny, nx)
print(image)
print("Shape:", image.shape)
print("Total photons:", image.sum())
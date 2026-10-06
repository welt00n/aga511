from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def radial_profile(frame):
	ny, nx = frame.shape

	# Pixel-center coordinates, mapped to [-1, 1]
	y, x = np.indices((ny, nx))
	x = (x + 0.5) / nx * 2.0 - 1.0
	y = (y + 0.5) / ny * 2.0 - 1.0

	radius = np.sqrt(x**2 + y**2)

	# Only consider pixels inside the stellar disk
	valid = radius <= 1.0

	# Radial bins
	bins = np.linspace(0.0, 1.0, 51)
	bin_centers = (bins[:-1] + bins[1:]) / 2

	brightness = np.zeros(len(bin_centers))

	for i, (r0, r1) in enumerate(zip(bins[:-1], bins[1:])):
		mask = valid & (radius >= r0) & (radius < r1)

		if np.any(mask):
			brightness[i] = frame[mask].mean()

	return bin_centers, brightness

def make_analysis(image, config):
	fig, axes = plt.subplots(config['n_observers'], 2, figsize=(12,10))
	
	frame_list = []
	images = []
	
	radial_lines = []
	radial_list = []

	for angle, (image_ax, radial_ax) in zip(range(config['n_observers']), axes):
		frames = image[:, angle].astype(np.float64)
		vmax = frames.max()
		frames = frames
		frame_list.append(frames)
		im = image_ax.imshow(
			frames[0],
			cmap="hot",
			vmin=0,
			vmax=vmax
		)
		images.append(im)
		
		radii = []
		brightnesses = []

		for frame in frames:
			r, brightness = radial_profile(frame)
			radii.append(r)
			brightnesses.append(brightness)

		radial_list.append((radii, brightnesses))

		line, = radial_ax.plot(radii[0], brightnesses[0])
		radial_lines.append(line)

		radial_ax.set_title(f"Observer imu={angle}")
		radial_ax.grid(True)

		radial_ax.set_xlim(0, 1)
		radial_ax.set_xlabel("Projected radius")
		
		radial_ax.set_ylim(0, max(brightnesses[0].max(), 1))
		radial_ax.set_ylabel("Brightness")
		
	def update(frame):
		for i, (im, frames) in enumerate(zip(images, frame_list)):
			im.set_data(frames[frame])
			radii, brightnesses = radial_list[i]
			radial_lines[i].set_data(radii[frame], brightnesses[frame])
		return images + radial_lines

	animation = FuncAnimation(fig, update, frames=image.shape[0], interval=1000 / 3)
	prefix = f"{datetime.now()}"
	prefix+= f"-{config['N']}-{config['nphi']}-{config['nmu']}-{config['ny']}-{config['nx']}"
	filename = prefix + "-Star.gif"
	animation.save(filename, writer=PillowWriter(fps=30))
	print("plt.show()")

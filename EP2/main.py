import os
import time
from datetime import datetime
import argparse
import subprocess
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def init():
	global ARGS
	global ENGINES
	ARGS = parse_args()
	config = {
		"N": 10 ** 20,
		"seed": 2,
		"nphi": 50,
		"nmu": 50,
		"nx": 100,
		"ny": 100,
	}
	config = {
		"N": 10 ** 10,
		"seed": 2,
		"nphi": 30,
		"nmu": 10,
		"nx": 100,
		"ny": 100,
	}
	return config

def parse_args():
	parser = argparse.ArgumentParser(description="AGA0511 EPs")
	parser.add_argument(
		'--experiment',
		type=str,
		default="star",
		choices=['buffon', 'star'],
		help="Choose either Buffon's needle Monte Carlo Experiment (buffon) or Star simulation (star)"
	)
	parser.add_argument(
		"--range",
		nargs=2,
		type=int,
		default=[6,10],
		metavar=("min", "max"),
		help="Minimum and maximum powers of 10 for N"
	)
	parser.add_argument(
		"--repetitions",
		type=int,
		default=5,
		help="Number of independent experiments for each N"
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Global seed to generate the experimental seeds"
	)
	parser.add_argument(
		"--engine",
		type=str,
		default="single-thread-c",
		help="Threadpool for faster experimentation, single-thread for single thread performance analysis",
		choices=("single-thread-c", "thread-pool-c")
	)
	return parser.parse_args()

def welcome_message():
	print(f"Starting experiment")

def run_star_c_engine(config):
	configs = [str(item) for item in config.values()]
	result = subprocess.run(
		["./star", *configs],
		capture_output=True,
		text=True,
		cwd='engine'
	)
	print(result.stdout)
	if result.returncode!=0:
		raise RuntimeError(f"Error running C engine:\n{result.stderr}")

def compile_star_c_engine():
	cwd = os.getcwd()
	engine_path = os.path.join(cwd, "engine")
	c_files = [item for item in os.listdir(engine_path) if item[-2:] == '.c']
	result = subprocess.run(
		["gcc", *c_files, "-o", "star", "-O2", "-lm"],
		capture_output=True,
		text=True,
		check=False,
		cwd=engine_path
	)
	if result.returncode != 0:
		raise RuntimeError(
			f"Error compiling C engine:\n{result.stderr}"
		)

def run_star(config):
	compile_star_c_engine()
	run_star_c_engine(config)

def finish_message():
	print("\nDone.")

def build_gif(config):
	bins = [item for item in os.listdir('engine') if item.endswith('.bin')]
	if not bins:
		raise Exception("Skipping build gif, couldnt find a bin file.")

	last_image_path = max(bins, key=lambda filename: os.path.getmtime(os.path.join('engine', filename)))

	data = np.fromfile(f'engine/{last_image_path}', dtype=np.uint32)
	
	expected_size = config["nphi"] * config["nmu"] * config["ny"] * config["nx"]
	if data.size != expected_size:
		raise ValueError(f"Expected {expected_size} values, but got {data.size}, review the binary image saving and loading process")

	image = data.reshape(config["nphi"], config["nmu"] , config["ny"] , config["nx"])
	
	draw_image(image, config)

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

def draw_image(image, config):
	angles = range(0, image.shape[1] if image.shape[1] <5 else 5)
	angles = [6, 8, 9, 10, 14]
	angles = [ 10, 14, 20, 24, 28, 33, 44]
	angles = [0, 1]
	angles = [0, 1, 2]
	angles = [4, 6, 8]
	fig, axes = plt.subplots(
		len(angles),
		2,
		figsize=(12,10)
	)
	images = []
	radial_lines = []
	frame_list = []
	radial_list = []

	for angle, (image_ax, radial_ax) in zip(angles, axes):

		# IMPORTANT: keep the original data
		raw_frames = image[:, angle].astype(np.float64)

		# -------------------------------------------------
		# LEFT: your current visualization
		# -------------------------------------------------

		vmax = raw_frames.max()

		# Your current inversion
		frames = raw_frames

		frame_list.append(frames)

		im = image_ax.imshow(
			frames[0],
			cmap="hot",
			vmin=0,
			vmax=vmax
		)

		images.append(im)

		# -------------------------------------------------
		# RIGHT: raw brightness vs radius
		# -------------------------------------------------

		radii = []
		brightnesses = []

		for frame in raw_frames:
			r, brightness = radial_profile(frame)
			radii.append(r)
			brightnesses.append(brightness)

		radial_list.append((radii, brightnesses))

		line, = radial_ax.plot(
			radii[0],
			brightnesses[0]
		)

		radial_lines.append(line)

		radial_ax.set_xlim(0, 1)
		radial_ax.set_ylim(
			0,
			max(brightnesses[0].max(), 1)
		)

		radial_ax.set_xlabel("Projected radius")
		radial_ax.set_ylabel("Brightness")
		radial_ax.set_title(f"Observer imu={angle}")

		radial_ax.grid(True)

	# -------------------------------------------------
	# Animation
	# -------------------------------------------------

	def update(frame):
		for i, (im, frames) in enumerate(zip(images, frame_list)):

			# Update image
			im.set_data(frames[frame])

			# Update radial profile
			radii, brightnesses = radial_list[i]

			radial_lines[i].set_data(
				radii[frame],
				brightnesses[frame]
			)

		return images + radial_lines

	animation = FuncAnimation(
		fig,
		update,
		frames=image.shape[0],
		interval=1000 / 3
	)

	animation.save(
		f"{datetime.now()}-"
		f"{config['N']}, "
		f"{config['nphi']}, "
		f"{config['nmu']}, "
		f"{config['ny']}, "
		f"{config['nx']}-Star.gif",
		writer=PillowWriter(fps=30)
	)

	plt.show()

if __name__ == '__main__':
	config = init()
	welcome_message()
	start = time.perf_counter()
	results = run_star(config)
	total_elapsed = time.perf_counter() - start
	build_gif(config)
	print(f"Finished in {total_elapsed}s.")
	finish_message()
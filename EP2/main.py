import copy
import os
import time
from datetime import datetime
import argparse
import subprocess
import numpy as np

from analysis import make_analysis, save_time_runtime_graph

def init():
	global ARGS
	ARGS = parse_args()
	# Order matters here, so, use modern python else we might have problems
	config = {
		"N": 10 ** min(ARGS.range),
		"seed": ARGS.seed,
		"a": ARGS.limb_darkening,
		"nspots": ARGS.nspots,
		"n_observers":ARGS.n_observers,
		"nx": ARGS.nx,
		"ny": ARGS.ny,
		"nphi": ARGS.nphi,
		"nmu": ARGS.nmu,
	}
	return config

def parse_args():
	parser = argparse.ArgumentParser(description="AGA0511 EP2")
	parser.add_argument(
		"--range",
		nargs=2,
		type=int,
		default=[6,12],
		metavar=("min", "max"),
		help="Minimum and maximum powers of 10 for N"
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Global seed to generate the experimental seeds"
	)
	parser.add_argument(
		"--limb_darkening",
		type=float,
		default=1.0,
		help="Parameter 'a' for the emission direction that generates the limb darkening effect"
	)
	parser.add_argument(
		"--nspots",
		type=int,
		default=3,
		help="Number of random solar spots to be on the star."
	)
	parser.add_argument(
		"--n_observers",
		type=int,
		default=4,
		help="Number of observers to capture photons from. We distribute the number of observers along imu."
	)
	parser.add_argument(
		"--nx",
		type=int,
		default=200,
		help="x axis output image resolution"
	)
	parser.add_argument(
		"--ny",
		type=int,
		default=200,
		help="y axis output image resolution"
	)
	parser.add_argument(
		"--nphi",
		type=int,
		default=150,
		help="Number of divisions on the observer sphere along the longitude. Defines the resolution of the observer (sun spots look less blurred on higher resolution)"
	)
	parser.add_argument(
		"--nmu",
		type=int,
		default=150,
		help="Number of divisions on the observer sphere along the latitude. Defines the resolution of the observer (sun spots look less blurred on higher resolution)"
	)
	return parser.parse_args()

def welcome_message():
	print(f"Starting experiment")

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
		raise RuntimeError(f"Error compiling C engine:\n{result.stderr}")

def run_star_c_engine(config):
	configs = [str(item) for item in config.values()]
	command = ["./star", *configs]
	print(" ".join(command))
	result = subprocess.run(command,cwd="engine")
	if result.returncode != 0:
		raise RuntimeError(f"C engine exited with code {result.returncode}")

def run_star(config):
	compile_star_c_engine()
	run_star_c_engine(config)

def get_image_data(config):
	bins = [item for item in os.listdir('engine') if item.endswith('.bin')]
	if not bins:
		raise Exception("Skipping build gif, couldnt find a bin file.")

	last_image_path = max(bins, key=lambda filename: os.path.getmtime(os.path.join('engine', filename)))

	data = np.fromfile(f'engine/{last_image_path}', dtype=np.uint32)
	
	expected_size = config["nphi"] * config["n_observers"] * config["ny"] * config["nx"]
	if data.size != expected_size:
		raise ValueError(f"Expected {expected_size} values, but got {data.size}, review the binary image saving and loading process")

	image = data.reshape(config["nphi"], config["n_observers"] , config["ny"] , config["nx"])
	return image

def clean_raw_data():
	bins = [item for item in os.listdir('engine') if item.endswith('.bin')]
	os.remove(f'engine/{bins[-1]}')

def build_simulation_analysis(config, run_id):
	image = get_image_data(config)
	make_analysis(image, config, run_id)
	clean_raw_data()

def finish_message():
	print("\nDone.")

if __name__ == '__main__':
	config = init()
	welcome_message()
	start = time.perf_counter()
	run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	print(start, "start", run_id)
	check_points = []
	os.mkdir(f"./{run_id}")
	for i in range(ARGS.range[0], ARGS.range[1]+1):
		config['N'] = 10 ** i
		run_start_at = time.perf_counter()
		run_star(config)
		run_time = time.perf_counter() - run_start_at
		check_points.append({'N':config['N'],'T':run_time})
		build_simulation_analysis(config, run_id)

	save_time_runtime_graph(check_points, run_id)
	total_elapsed = time.perf_counter() - start 
	print(f"Finished in {total_elapsed}s. Check output in: {os.getcwd()}/{run_id}")
	finish_message()

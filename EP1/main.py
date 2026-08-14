import time
import math
from datetime import datetime

from tqdm import tqdm

from matplotlib import pyplot as plt
import random
import numpy as np
import torch

from utils import get_env

device = get_env('device', 'cpu')
SEED = get_env('SEED', 42)
VERBOSE=get_env('verbose', False, bool)
N = get_env('N', 5e2, int)
L = 1.0
D = 1.0

class DeviceMapping:
    """Just some over-engeneering to pass the time... """
    def __init__(self, device):
        self.device = device
        self.device_mapping = {
                'cpu': {
                    'sin': math.sin,
                    'pi': math.pi
                },
                'numpy': {
                    'sin': np.sin,
                    'pi': np.pi
                },
                'torch': {
                    'sin': torch.sin,
                    'pi': torch.pi
                }
            }

    @property
    def sin(self):
        return self.device_mapping[self.device]['sin']

    @property
    def pi(self):
        return self.device_mapping[self.device]['pi']

mappings = DeviceMapping(device)

pi = mappings.pi
sin = mappings.sin

def generate_toss_data(x_range, theta_range):    
    x = random.uniform(0, 1) * x_range
    theta = random.uniform(0,1) * theta_range
    return x, theta

def test_needle_cross(x, theta):
    return x <= (L/2) * sin(theta)

def toss():
    x, theta = generate_toss_data(D/2, pi/2)
    return test_needle_cross(x, theta)

def welcome_message():
    print(f'Starting')    

def init():
    print(f"Initiating process for N={N:.2e} on the {device}...")

def run(n, steps=1, verbose=False, progress=True):
    """
    What does it mean to run? I want to make it some sort of cli 
    or at least a nice script with some cool entrypoints or/and docs for configuration. 
    Lets start with some basic running for now, we will later have simulations and analysis and much more.


    -- aug 14
    well, runing now means simulating N times and capturing results every k steps so that we can plot the progress later
    """
    results = []
    random.seed(SEED)
    count = 0
    iterator = tqdm(range(n), desc='Running...') if progress else range(n)
    start_time = time.perf_counter()
    segment_size = n//steps
    for i in iterator:
        res = toss()
        if res:
            count+=1

        cycles = i+1
        if cycles % segment_size == 0:
            results.append({
                'n': cycles,
                'pi': 2*(cycles/count),
                'elapsed_time': time.perf_counter() - start_time,
            })

    return results

def finish_message():
    print("\nDone.")

if __name__ == '__main__':
    welcome_message()
    init()
    results = []
    steps = get_env("STEPS", 2, int)
    start = time.perf_counter()
    results = run(N, steps=steps)
    total_elapsed = time.perf_counter() - start    

    n = np.array([result['n'] for result in results])
    estimated_pi = np.array([result['pi'] for result in results])
    abs_error = np.abs(estimated_pi - math.pi)
    elapsed_time = np.array([result['elapsed_time'] for result in results])
    iterations_sec = n/elapsed_time
    target_error = 1e-14

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    ax = axes[0]
    ax.plot(n, estimated_pi, label='Estimated pi')
    ax.axhline(pi, linestyle='--', label='Reference pi')

    ax.set_title('Estimated pi vs N')
    ax.set_xlabel("N")
    ax.set_ylabel("Estimated pi")
    
    ax.legend() # test it
    ax.grid(True)


    ax = axes[1]
    ax.plot(n, abs_error)
    ax.set_yscale('log') # In case it is difficult to see since the y set may be huge in scale varying many orders of magnitude
    ax.axhline(y=target_error, color="red", linestyle="--", label=f"Target = {target_error:.2}")

    ax.set_title("Absolute error vs N")
    ax.set_xlabel("N")
    ax.set_ylabel("AbsError: |Estimated pi-pi|")
    ax.legend()
    ax.grid(True)

    ax = axes[2]
    ax.plot(elapsed_time, n)
    ax.set_title("N vs Elapsed Time")
    ax.set_xlabel("Elapsed time")
    ax.set_ylabel("N")
    ax.grid(True)

    plt.tight_layout()
    filename = datetime.now().strftime("%d-%m-%Y-%H:%M:%S-results.png")
    fig.savefig(filename, dpi=300, bbox_inches="tight")    
    finish_message()
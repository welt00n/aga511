import time
import math

from tqdm import tqdm

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

def run(n, verbose=False, progress=True):
    """
    What does it mean to run? I want to make it some sort of cli 
    or at least a nice script with some cool entrypoints or/and docs for configuration. 
    Lets start with some basic running for now, we will later have simulations and analysis and much more.
    """
    random.seed(SEED)
    count = 0
    iterator = tqdm(range(n), desc='Running...') if progress else range(n)
    for i in iterator:
        res = toss()
        if res:
            count+=1
    if verbose:
        print(f"Finished with count: {count}")

    pi_ = 2*(n/count)
    if verbose:
        print(f"Approximated pi as : {pi_}")

        print("Running some statistics...")

        print(f"The difference to the reference value: {np.abs(pi_ - pi)}")
    return pi_

def finish_message():
    print("\nDone.")

if __name__ == '__main__':
    welcome_message()
    init()
    results = []
    steps = get_env("STEPS", 2, int)
    for i in range(1, steps+1):
        if i==0:
            continue
        n = int(N/steps)
        start = time.perf_counter()
        result = run(n * i)
        time_elapsed = time.perf_counter() - start
        results.append(result)

    print(f"Reference Value: {pi}")
    for result in results:
        print(result)


    finish_message()
import json
import os
import socket
import statistics

SPEEDS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "speeds.json")
LAPTOP_GUESSES = {"cpu64": 8.0e6, "cpu32": 1.2e7, "gpu64": 6.0e8, "gpu32": 9.0e9}
LAPTOP_SMS = 36
MINIMUM_SECONDS = 0.5
KEPT = 5


def binary_key(processor, precision):
    return f"{processor}{precision}"


def read_speeds():
    if not os.path.exists(SPEEDS):
        return {}
    with open(SPEEDS) as file:
        return json.load(file)


def write_speeds(speeds):
    os.makedirs(os.path.dirname(SPEEDS), exist_ok=True)
    partial_path = SPEEDS + ".partial"
    with open(partial_path, "w") as file:
        json.dump(speeds, file, indent=2)
    os.replace(partial_path, SPEEDS)


def remember_speed(key, photons_per_second, units):
    speeds = read_speeds()
    host = speeds.setdefault(socket.gethostname(), {})
    entry = host.setdefault(key, {"photons_per_second": [], "units": units})
    entry["photons_per_second"] = (entry["photons_per_second"] + [photons_per_second])[-KEPT:]
    entry["units"] = units
    write_speeds(speeds)


def remember_engine_run(summary):
    on_every_unit = summary["COMPUTE_UNITS"] == summary["COMPUTE_UNITS_TOTAL"]
    if summary["PHOTONS_TIME"] >= MINIMUM_SECONDS and on_every_unit:
        key = binary_key(summary["PROCESSOR"].lower(), summary["PRECISION"])
        remember_speed(key, summary["PHOTONS_PER_SECOND"], summary["COMPUTE_UNITS_TOTAL"])


def measured_speed(key):
    entry = read_speeds().get(socket.gethostname(), {}).get(key)
    if not entry or not entry["photons_per_second"]:
        return None
    return statistics.median(entry["photons_per_second"]), entry["units"]


def speed_of(processor, precision, sms=None):
    key = binary_key(processor, precision)
    measured = measured_speed(key)
    photons_per_second, units = measured or (LAPTOP_GUESSES[key], LAPTOP_SMS if processor == "gpu" else 1)
    if processor == "gpu" and sms:
        photons_per_second *= min(sms, units) / units
    return photons_per_second, "measured here" if measured else "the laptop's figure (not measured here)"

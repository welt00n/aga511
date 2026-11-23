import functools
import json
import math
import sys

from lab import engine, prompts, runs, speeds

AT_LEAST_SECONDS = 1.0
AT_LEAST_CHUNKS_PER_THREAD = 32
SERIAL_SECONDS = 0.6
DISK_BYTES_PER_SECOND = 1e9
NICE_STEPS = (1, 1.5, 2, 3, 4, 5, 6, 8, 10)

MINIMUM_RULE = (
    "A run measures speed only if the GPU is busy long enough and evenly enough. Two rules, the larger wins: "
    f"(1) at least {AT_LEAST_SECONDS:g} s of photons, so that the launch and the clock's ramp-up don't dominate; "
    f"(2) at least {AT_LEAST_CHUNKS_PER_THREAD} chunks per resident thread, so that every SM gets a fair share and the "
    "slowest one doesn't decide the time (at 10⁷ photons on the laptop: 9 766 chunks for 27 648 threads, and pass "
    "times varied 0.6–1.6×; research/pass_order.py).")


# ============================================================ a point: one engine run

def point(processor, precision, photons, sms=None, variant=None, repeat=1, **star):
    return {"processor": processor, "precision": precision, "sms": sms, "photons": int(photons),
            "variant": variant, "repeat": repeat, "star": star}


def key(each):
    name = f"{each['processor']}{each['precision']}"
    name += f"-{each['variant']}" if each["variant"] else ""
    name += f"-sms{each['sms']:02d}" if each["sms"] else ""
    name += f"-n{each['photons']:.2e}-r{each['repeat']}"
    if each["star"]:
        name += "-" + "-".join(f"{option}{value}" for option, value in sorted(each["star"].items()))
    return name.replace("+", "")


def star_config(each):
    return engine.default_star_config(each["photons"], sms=each["sms"], variant=each["variant"], **each["star"])


# ============================================================ what the engine says before running

@functools.cache
def described(processor, precision, sms, variant, star_json):
    config = engine.default_star_config(1, sms=sms, variant=variant, **json.loads(star_json))
    return engine.describe_engine(config, processor, precision)[0]


def describe(each):
    return described(each["processor"], each["precision"], each["sms"], each["variant"],
                     json.dumps(each["star"], sort_keys=True))


def resident_threads(launch):
    if launch["PROCESSOR"] != "GPU":
        return launch["COMPUTE_UNITS"]
    return launch["COMPUTE_UNITS"] * launch["THREADS_PER_BLOCK"] * launch["BLOCKS_PER_UNIT"]


# ============================================================ the minimum N, and why

def minimum_photons(processor, precision, sms=None, variant=None):
    launch = described(processor, precision, sms, variant, "{}")
    speed, source = speeds.speed_of(processor, precision, sms)
    by_time = AT_LEAST_SECONDS * speed
    by_work = AT_LEAST_CHUNKS_PER_THREAD * resident_threads(launch) * launch["CHUNK_SIZE"]
    return {"minimum": int(max(by_time, by_work)), "by_time": int(by_time), "by_work": int(by_work),
            "speed": speed, "speed_source": source, "resident_threads": resident_threads(launch)}


def nice_above(value):
    exponent = math.floor(math.log10(value))
    for step in NICE_STEPS:
        if step * 10 ** exponent >= value:
            return int(step * 10 ** exponent)


def reference_photons(processor, precision, seconds):
    minimum = minimum_photons(processor, precision)
    return nice_above(max(minimum["minimum"], seconds * minimum["speed"]))


def validity(each):
    if each["photons"] >= minimum_photons(each["processor"], each["precision"], each["sms"])["minimum"]:
        return "valid"
    return "INVALID"


# ============================================================ predicting a point

def predict(each):
    speed, source = speeds.speed_of(each["processor"], each["precision"], each["sms"])
    launch = describe(each)
    disk_bytes = launch["OBSERVER_BYTES"] * launch["OBSERVERS"]
    seconds = launch["PASSES"] * each["photons"] / speed + SERIAL_SECONDS + disk_bytes / DISK_BYTES_PER_SECOND
    return {"seconds": seconds, "speed": speed, "speed_source": source, "passes": launch["PASSES"],
            "device_bytes": launch["IMAGE_BYTES"], "disk_bytes": disk_bytes}


# ============================================================ a plan: every point, each once

class Plan:
    def __init__(self):
        self.entries = {}
        self.tasks = []

    def add(self, each, part):
        name = key(each)
        if name not in self.entries:
            self.entries[name] = {"key": name, "point": each, "parts": [], "prediction": predict(each),
                                  "validity": validity(each)}
        if part not in self.entries[name]["parts"]:
            self.entries[name]["parts"].append(part)
        return name

    def add_task(self, part, name, seconds, detail):
        self.tasks.append({"part": part, "name": name, "seconds": seconds, "detail": detail})

    def keys_of(self, part):
        return [name for name, entry in self.entries.items() if part in entry["parts"]]

    def points(self):
        return list(self.entries.values())

    def total_seconds(self):
        return (sum(entry["prediction"]["seconds"] for entry in self.entries.values())
                + sum(task["seconds"] for task in self.tasks))

    def invalid(self):
        return [entry for entry in self.entries.values() if entry["validity"] == "INVALID"]

    def shared(self):
        return [entry for entry in self.entries.values() if len(entry["parts"]) > 1]

    def as_data(self):
        return {"points": self.points(), "tasks": self.tasks, "total_seconds": self.total_seconds(),
                "minimum_rule": MINIMUM_RULE}

    def save(self, path):
        runs.save_json(self.as_data(), path)


# ============================================================ showing it, and asking

def duration(seconds):
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def part_rows(plan):
    parts = list(dict.fromkeys([part for entry in plan.points() for part in entry["parts"]] +
                               [task["part"] for task in plan.tasks]))
    rows = []
    for part in parts:
        entries = [entry for entry in plan.points() if part in entry["parts"]]
        own = [entry for entry in entries if entry["parts"][0] == part]
        tasks = [task for task in plan.tasks if task["part"] == part]
        photons = [entry["point"]["photons"] for entry in entries]
        verdicts = sorted({entry["validity"] for entry in entries})
        seconds = sum(entry["prediction"]["seconds"] for entry in own) + sum(task["seconds"] for task in tasks)
        span = f"{min(photons):.1e} – {max(photons):.1e}" if photons else "—"
        work = f"{len(entries)} runs" + (f" ({len(entries) - len(own)} shared)" if len(own) < len(entries) else "")
        rows.append((part, work + (f" + {len(tasks)} checks" if tasks else ""), span,
                     ", ".join(verdicts) or "—", duration(seconds)))
    return rows


def plan_lines(plan):
    rows = [("part", "work", "N", "validity", "predicted")] + part_rows(plan)
    widths = [max(len(row[column]) for row in rows) for column in range(5)]
    lines = ["  " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths)) for row in rows]
    lines.insert(1, "  " + "  ".join("-" * width for width in widths))
    peak = max([entry["prediction"]["device_bytes"] for entry in plan.points()] or [0])
    disk = max([entry["prediction"]["disk_bytes"] for entry in plan.points()] or [0])
    lines += ["", f"  total: {duration(plan.total_seconds())}; {len(plan.points())} engine runs "
                  f"({len(plan.shared())} shared between parts, run once); at most {peak / 1e9:.2f} GB on the "
                  f"device and {disk / 1e9:.2f} GB on disk at a time"]
    sources = sorted({entry["prediction"]["speed_source"] for entry in plan.points()})
    lines.append("  speeds: " + "; ".join(sources))
    for task in plan.tasks:
        lines.append(f"  check {task['name']}: {task['detail']}")
    return lines


def minimum_lines(binaries):
    lines = ["", "  The minimum N for a valid measurement on this machine:"]
    for processor, precision, sms in binaries:
        minimum = minimum_photons(processor, precision, sms)
        on = f"{sms} SM" if sms else "all SMs"
        lines.append(f"    {processor}{precision} on {on}: {minimum['minimum']:.1e} = max({minimum['by_time']:.1e} for "
                     f"{AT_LEAST_SECONDS:g} s at {minimum['speed']:.2e}/s, {minimum['by_work']:.1e} for "
                     f"{AT_LEAST_CHUNKS_PER_THREAD} chunks × {minimum['resident_threads']} resident threads)")
    return lines


def show(title, plan, binaries=()):
    print(f"\n{title}\n")
    print("\n".join(plan_lines(plan) + (minimum_lines(binaries) if binaries else [])))
    invalid = plan.invalid()
    if invalid:
        print(f"\n  WARNING: {len(invalid)} runs are below the minimum N, so this is NOT a valid measurement.")
        print("  " + MINIMUM_RULE)


def confirmed(yes):
    if yes:
        return True
    if not sys.stdin.isatty():
        print("\nNot asked (no terminal): run again with --yes to go ahead.")
        return False
    return prompts.ask_yes("\nGo ahead?", default=True)

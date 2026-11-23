import glob
import json
import os
import socket
import traceback
from datetime import datetime

from lab import engine

RUNS_FOLDER = os.path.join(engine.EP4_FOLDER, "runs")
INDEX = os.path.join(RUNS_FOLDER, "index.md")
STAMP = "%Y-%m-%d_%H-%M-%S"
COMMAND = []


# ============================================================ json, written whole or not at all

def plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON")


def save_json(data, path):
    partial_path = path + ".partial"
    with open(partial_path, "w") as file:
        json.dump(data, file, indent=2, default=plain)
    os.replace(partial_path, path)


def read_json(path):
    with open(path) as file:
        return json.load(file)


def save_text(lines, path):
    with open(path, "w") as file:
        file.write("\n".join(lines) + "\n")


def now():
    return datetime.now().isoformat(timespec="seconds")


def remember_command(argv):
    COMMAND[:] = ["python", "main.py", *argv]


def machine_of(summary):
    name = summary["PROCESSOR_NAME"].replace("_", " ")
    if summary["PROCESSOR"] == "GPU":
        return f"{name} (cc {summary['COMPUTE_CAPABILITY']}, {summary['COMPUTE_UNITS_TOTAL']} SMs)"
    return f"{name} (the sequential reference)"


# ============================================================ a run: runs/<kind>/<stamp>-<label>/

class Run:
    def __init__(self, kind, label, asked, folder=None):
        self.folder = folder or os.path.join(RUNS_FOLDER, kind, datetime.now().strftime(STAMP) + "-" + label)
        self.record_path = os.path.join(self.folder, "run.json")
        self.record = read_json(self.record_path) if folder else {
            "kind": kind, "label": label, "command": COMMAND, "asked": asked, "code_hash": engine.code_hash(),
            "host": socket.gethostname(), "machine": [], "started": now(), "finished": None,
            "status": "running", "passed": None, "headline": None}

    @classmethod
    def reopen(cls, folder):
        return cls(None, None, None, os.path.abspath(folder))

    def __enter__(self):
        os.makedirs(self.folder, exist_ok=True)
        self.record.update(status="running", finished=None)
        save_json(self.record, self.record_path)
        return self

    def __exit__(self, kind, error, trace):
        if error is None:
            return False
        stopped = isinstance(error, KeyboardInterrupt)
        self.record.update(status="interrupted" if stopped else "failed", finished=now(),
                           headline=self.record["headline"] or ("interrupted" if stopped else f"failed: {error}"))
        save_json(self.record, self.record_path)
        details = [] if stopped else ["```", traceback.format_exc().rstrip(), "```"]
        save_text(summary_lines(self.record, details, [], self.folder), os.path.join(self.folder, "summary.md"))
        write_index()
        return False

    def path(self, *parts):
        return os.path.join(self.folder, *parts)

    def add_machine(self, summary):
        machine = machine_of(summary)
        if machine not in self.record["machine"]:
            self.record["machine"].append(machine)

    def finish(self, headline, results, figures=(), passed=None):
        self.record.update(status="done", finished=now(), headline=headline,
                           passed=None if passed is None else bool(passed))
        save_json(self.record, self.record_path)
        save_text(summary_lines(self.record, results, figures, self.folder), self.path("summary.md"))
        write_index()
        print(f"\n{headline}\n  {os.path.relpath(self.path('summary.md'), engine.EP4_FOLDER)}")


# ============================================================ summary.md: the run's page

def verdict(record):
    return {True: "passed", False: "FAILED"}.get(record["passed"], record["status"])


def asked_rows(asked):
    return [f"| {key} | {value} |" for key, value in asked.items() if value is not None]


def file_list(folder):
    names = sorted(os.listdir(folder))
    return [f"- [`{name}/`]({name}/)" if os.path.isdir(os.path.join(folder, name)) else f"- [`{name}`]({name})"
            for name in names if not name.endswith(".partial")]


def summary_lines(record, results, figures, folder):
    machine = "; ".join(record["machine"]) or record["host"]
    lines = [f"# {record['kind'].capitalize()}: {record['label']}", "",
             f"{record['started']} · {machine} · code `{record['code_hash']}` · {verdict(record)}", "",
             f"**{record['headline']}**", "", f"`{' '.join(record['command'])}`", "",
             "## What was asked", "", "| option | value |", "|---|---|", *asked_rows(record["asked"]), "",
             "## Results", "", *results]
    if figures:
        lines += ["", "## Figures", ""] + [f"![{os.path.basename(path)}]({path})" for path in figures]
    return lines + ["", "## Files", "", *file_list(folder)]


# ============================================================ the index of every run

def run_records():
    records = []
    for path in glob.glob(os.path.join(RUNS_FOLDER, "*", "*", "run.json")):
        record = read_json(path)
        record["folder"] = os.path.relpath(os.path.dirname(path), RUNS_FOLDER)
        records.append(record)
    return sorted(records, key=lambda record: record["started"], reverse=True)


def index_lines(records):
    lines = ["# Runs", "", "Newest first. Each run's page is its `summary.md`.", "",
             "| started | kind | run | result | headline |", "|---|---|---|---|---|"]
    for record in records:
        lines.append(f"| {record['started'].replace('T', ' ')} | {record['kind']} | "
                     f"[{record['label']}]({record['folder']}/summary.md) | {verdict(record)} | "
                     f"{record['headline'] or ''} |")
    return lines


def write_index():
    os.makedirs(RUNS_FOLDER, exist_ok=True)
    lines = index_lines(run_records())
    save_text(lines, INDEX)
    return lines

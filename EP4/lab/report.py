import os
import shutil
import subprocess
import sys
import tempfile

import nbformat
from nbclient import NotebookClient

from lab import environment

KERNEL = "python3"
TIMEOUT_SECONDS = 600
PREAMBLE_FIXES = "\\newcounter{none}\n"
HELPERS = '''import json
import os

from IPython.display import Image, Markdown, display


def load(name):
    with open(os.path.join("parts", name + ".json")) as file:
        return json.load(file)


def cell(value, form):
    if value is None:
        return "—"
    if callable(form):
        return form(value)
    return format(value, form) if form else str(value)


def table(rows, columns):
    """columns: (header, a key or a function of the row, a format spec or a function)"""
    def value(row, pick):
        return pick(row) if callable(pick) else row.get(pick)
    lines = ["| " + " | ".join(header for header, _, _ in columns) + " |",
             "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(cell(value(row, pick), form) for _, pick, form in columns) + " |")
    display(Markdown("\\n".join(lines)))


def figure(path):
    display(Image(filename=path, width=720))
'''


def markdown(text):
    return nbformat.v4.new_markdown_cell(text.strip("\n"))


def code(source):
    return nbformat.v4.new_code_cell(source.strip("\n"))


def notebook(title, cells):
    book = nbformat.v4.new_notebook()
    book.metadata["title"] = title
    book.metadata["kernelspec"] = {"name": KERNEL, "display_name": "Python 3", "language": "python"}
    book.cells = [code(HELPERS), *cells]
    return book


def execute(book, folder):
    NotebookClient(book, timeout=TIMEOUT_SECONDS, kernel_name=KERNEL,
                   resources={"metadata": {"path": folder}}).execute()
    return book


def with_environment_tools():
    tools = os.path.dirname(sys.executable)
    if tools not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] = tools + os.pathsep + os.environ["PATH"]


def latex_of(book):
    with_environment_tools()
    from nbconvert import LatexExporter
    exporter = LatexExporter()
    exporter.exclude_input = True
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, resources = exporter.from_notebook_node(book, resources={"output_files_dir": "report_files"})
    start = body.index("\n", body.index("\\documentclass")) + 1
    return body[:start] + PREAMBLE_FIXES + body[start:], resources.get("outputs", {})


def pdf(book, path):
    tectonic = environment.tool("tectonic")
    if not tectonic:
        raise RuntimeError("tectonic not found")
    body, outputs = latex_of(book)
    with tempfile.TemporaryDirectory() as build:
        for name, data in outputs.items():
            os.makedirs(os.path.join(build, os.path.dirname(name)), exist_ok=True)
            with open(os.path.join(build, name), "wb") as file:
                file.write(data)
        with open(os.path.join(build, "report.tex"), "w") as file:
            file.write(body)
        result = subprocess.run([tectonic, "--keep-logs", "report.tex"], cwd=build, capture_output=True, text=True)
        if result.returncode != 0:
            errors = [line for line in result.stderr.splitlines() if line.startswith(("error", "!"))]
            raise RuntimeError("tectonic failed: " + "; ".join(errors[:4]))
        shutil.copy(os.path.join(build, "report.pdf"), path)
    return path


def html(book, path):
    from nbconvert import HTMLExporter
    exporter = HTMLExporter()
    exporter.exclude_input = True
    body, _ = exporter.from_notebook_node(book)
    with open(path, "w") as file:
        file.write(body)
    return path


def write_report(title, cells, folder, name="report"):
    book = execute(notebook(title, cells), folder)
    notebook_path = os.path.join(folder, name + ".ipynb")
    nbformat.write(book, notebook_path)
    try:
        document = pdf(book, os.path.join(folder, name + ".pdf"))
    except Exception as error:
        print(f"  the PDF failed ({error}); writing HTML instead", flush=True)
        document = html(book, os.path.join(folder, name + ".html"))
    return notebook_path, document

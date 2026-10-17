"""Reexecute this folder's notebook with genuine outputs.
Install nbformat, nbclient, ipykernel and the notebook's imported packages first.
Run: python rerun.py
Uses the current Python interpreter's kernel environment.
"""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

folder = Path(__file__).resolve().parent
path = folder / "notebook.ipynb"
nb = nbformat.read(path, as_version=4)
for cell in nb.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None
# Explicit kernel command makes the interpreter running this script reproducible.
km = KernelManager(kernel_name="python3")
km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
client = NotebookClient(nb, km=km, timeout=600, allow_errors=False,
                        resources={"metadata": {"path": str(folder)}})
try:
    client.execute()
finally:
    nbformat.write(nb, path)
print(f"Saved fresh outputs: {path}")

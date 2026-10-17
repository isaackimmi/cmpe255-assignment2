"""Re-execute this notebook locally; retain actual outputs and execution coverage.
Install requirements from the repository environment before running this file.
Run: python reproduce.py
"""
from pathlib import Path
import json, time, importlib.metadata
import nbformat
from nbclient import NotebookClient

folder = Path(__file__).resolve().parent
path = folder / "notebook.ipynb"
record_path = folder / "execution.json"
record = json.loads(record_path.read_text()) if record_path.exists() else {}
nb = nbformat.read(path, as_version=4)
for cell in nb.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None
start = time.time()
client = NotebookClient(nb, timeout=2400, kernel_name="python3", allow_errors=True,
                        resources={"metadata": {"path": str(folder)}})
try:
    client.execute()
finally:
    nbformat.write(nb, path)
    codes = [(i,c) for i,c in enumerate(nb.cells) if c.cell_type == "code"]
    record.update(code_cells=len(codes),
        executed_cells=sum(c.execution_count is not None for i,c in codes),
        elapsed_seconds=round(time.time()-start, 2),
        errors=[{"cell":i,"ename":o.ename,"evalue":o.evalue}
                for i,c in codes for o in c.outputs if o.output_type=="error"],
        environment={p:importlib.metadata.version(p) for p in
                     ["numpy","matplotlib","scipy","sympy","pandas","scikit-learn","nbclient","ipykernel"]},
        output_inventory={"png_outputs":sum("image/png" in o.get("data",{}) for i,c in codes for o in c.outputs),
                          "cells_with_output":sum(bool(c.outputs) for i,c in codes)})
    record_path.write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record, indent=2))
if record["errors"] or record["executed_cells"] != record["code_cells"]:
    raise SystemExit("Execution incomplete or errors present; inspect execution.json")

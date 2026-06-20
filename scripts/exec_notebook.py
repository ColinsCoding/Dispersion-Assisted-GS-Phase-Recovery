"""Execute a notebook in place, embedding outputs: py -3.13 scripts/exec_notebook.py <path>."""
import pathlib
import sys

import nbformat as nbf
from nbclient import NotebookClient

NB = pathlib.Path(sys.argv[1]).resolve()
if not NB.exists():
    raise SystemExit(f"no such notebook: {NB}")

nb = nbf.read(NB, as_version=4)
# kernel: explicit argv[2] > notebook's own kernelspec > "python3"
kernel = (sys.argv[2] if len(sys.argv) > 2
          else nb.metadata.get("kernelspec", {}).get("name", "python3"))
client = NotebookClient(nb, timeout=900, kernel_name=kernel,
                        resources={"metadata": {"path": str(NB.parent)}})
client.execute()
nbf.write(nb, NB)

n_err = sum(1 for c in nb.cells if c.cell_type == "code"
            for o in c.get("outputs", []) if o.get("output_type") == "error")
print(f"executed {NB.name}: {len(nb.cells)} cells, {n_err} errors")

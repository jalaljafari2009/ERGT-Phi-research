"""Generate the M0-B launcher notebook; no cloud execution occurs here."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ergt_phi.research_paths import workspace_path
ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


cells = [
    cell("markdown", """# ERGT-Phi M0-B: registered reference training

Select a CUDA GPU runtime. Upload **ERGT-Phi-M0.zip** produced by the research workspace.
The default is the unchanged registered single-seed validation protocol. It produces
a development reference, not a four-seed reproduction. Set MODE to 'four' only for
the full registered study; do not edit scientific seeds, thresholds or recipes.
Results and original resume checkpoints are written to your Google Drive.
The original checkpoint format is preserved. M0 research full-state resume tests
are a separate engineering check. No phase extension is activated.
"""),
    cell("code", """from pathlib import Path
import os, sys, zipfile
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
from google.colab import files
uploaded = files.upload()
archive_path = next((Path(name) for name in uploaded if Path(name).name == 'ERGT-Phi-M0.zip'), None)
if archive_path is None:
    raise RuntimeError('Upload ERGT-Phi-M0.zip from the M0 research workspace')
destination = Path('/content').resolve()
with zipfile.ZipFile(archive_path) as archive:
    for item in archive.infolist():
        target = (destination / item.filename).resolve()
        if destination != target and destination not in target.parents:
            raise RuntimeError('Unsafe archive member')
    archive.extractall(destination)
ROOT = destination / 'ERGT-Phi-research'
sys.path.insert(0, str(ROOT))
from ergt_phi.runtime import activate_reference
activate_reference()
from ergt_four_seed.environment import ensure_required_packages, verify_environment
from ergt_four_seed.integrity import verify_integrity
print(verify_integrity(ROOT / 'reference'))
print(ensure_required_packages(install_missing=True))
print(verify_environment())
import torch
if not torch.cuda.is_available():
    raise RuntimeError('Select a CUDA GPU runtime')
"""),
    cell("code", """from google.colab import drive
drive.mount('/content/drive')
import subprocess
MODE = 'single'
OUTPUT = Path('/content/drive/MyDrive/ERGT_Phi_M0/reference_training')
subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/run_reference_training.py'),
                '--mode', MODE, '--output', str(OUTPUT)], cwd=ROOT, check=True)
"""),
    cell("code", """import json
status = json.loads((ROOT / 'research/workspace/manifests/reference_training_status.json').read_text())
print(json.dumps(status, indent=2))
run_root = Path(status['run_root'])
print((run_root / 'final_verdict.json').read_text())
print('Keep the full run directory, checkpoint_manifest, protocol and environment records.')
"""),
]
target = workspace_path(ROOT, "notebook/M0_Reference_Training.ipynb")
target.write_text(json.dumps({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 4}, indent=2), encoding="utf-8")
for item in cells:
    if item["cell_type"] == "code":
        compile("".join(item["source"]), "colab_cell", "exec")
print(target)

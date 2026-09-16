"""Build a portable source/evidence archive; exclude venv and model weights."""
from pathlib import Path
import hashlib
import json
import zipfile
ROOT = Path(__file__).resolve().parents[1]
paths = []
for directory in ("reference", "ergt_phi", "scripts", "configs", "tests", "docs", "manifests", "notebook"):
    paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts)
paths += [ROOT / "README.md", ROOT / "requirements-m0.lock.txt", ROOT / ".gitignore"]
paths += list((ROOT / "runs/m0").glob("*.log")) + list((ROOT / "runs/m0").glob("*.xml")) + list((ROOT / "runs/m0").glob("*.json"))
target = ROOT.parent / "ERGT-Phi-M0.zip"
with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(set(paths)):
        archive.write(path, "ERGT-Phi-research/" + path.relative_to(ROOT).as_posix())
print(json.dumps({"archive": str(target), "bytes": target.stat().st_size,
    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "files": len(set(paths))}))

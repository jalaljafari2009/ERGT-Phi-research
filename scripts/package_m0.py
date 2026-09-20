"""Build the legacy M0 compatibility package; new research uses scripts/research.py."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    # Archived notebook stays byte-identical; only the working launcher changes.
    subprocess.run([sys.executable, "-B", str(ROOT / "scripts/build_colab_notebook.py")], cwd=ROOT, check=True)
    paths = []
    excluded_parts = {"__pycache__", ".pytest_cache", ".git", ".venv"}
    excluded_suffixes = {".pt", ".pth", ".zip", ".pyc", ".pem", ".key"}
    for directory in ("reference", "ergt_phi", "scripts", "configs", "tests", "docs", "research_tools", "research/legacy"):
        paths.extend(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and not excluded_parts.intersection(p.parts)
                     and p.suffix.lower() not in excluded_suffixes)
    paths += [ROOT / "README.md", ROOT / "AGENTS.md", ROOT / "requirements-m0.lock.txt", ROOT / ".gitignore",
              ROOT / "research/workspace/notebook/M0_Reference_Training.ipynb"]
    target = ROOT / "research/packages/legacy-M0/ERGT-Phi-M0.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            archive.write(path, "ERGT-Phi-research/" + path.relative_to(ROOT).as_posix())
    print(json.dumps({"archive": str(target), "bytes": target.stat().st_size,
                      "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "files": len(set(paths)),
                      "scope": "legacy_M0_compatibility_export_not_a_versioned_research_bundle"}))


if __name__ == "__main__":
    main()

"""Create local Git snapshots using Dulwich when the Git CLI is unavailable."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ergt_phi.research_paths import current_input, workspace_path
import json
from dulwich import porcelain
ROOT = Path(__file__).resolve().parents[1]
identity = b"ERGT M0 Research <m0@local>"
if not (ROOT / ".git").exists():
    repo = porcelain.init(ROOT)
    reference_paths = [p.relative_to(ROOT).as_posix() for p in (ROOT / "reference").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    reference_paths += [current_input(ROOT, "manifests/reference.json").relative_to(ROOT).as_posix(), "docs/MATHEMATICAL_SPEC.md", ".gitignore"]
    porcelain.add(repo, paths=reference_paths)
    first = porcelain.commit(repo, message=b"Pin unchanged ERGT reproduction reference and migration specification", author=identity, committer=identity)
    (workspace_path(ROOT, "manifests/version_control.json")).write_text(json.dumps({
        "reference_commit": first.decode(), "backend": "dulwich", "remote": None,
        "note": "Local repository only. No publication or external push.",
    }, indent=2), encoding="utf-8")
porcelain.add(str(ROOT))
commit = porcelain.commit(str(ROOT), message=b"Implement M0 native refactor, exact parity and full-state resume acceptance", author=identity, committer=identity)
print(commit.decode())
print(porcelain.status(str(ROOT)))

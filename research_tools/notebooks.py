"""Generate immutable, explained Colab notebooks and pinned source packages."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from research_tools.runner import safe_path, sha256


SOURCE_DIRECTORIES = ("ergt_phi", "reference", "scripts", "configs", "research_tools", "tests")
SOURCE_FILES = (
    "requirements-m0.lock.txt", "README.md", "AGENTS.md",
    "docs/MATHEMATICAL_SPEC.md", "docs/SPEC_Q_ANCHOR_ADDENDUM.md",
    "research/specification.lock.json",
    # Small accepted metadata needed by relocated legacy readers. Raw data and
    # checkpoints remain explicit, hash-pinned protocol inputs.
    "research/legacy/path_map.json",
    "research/legacy/LEGACY-M0/manifests/reference.json",
    "research/legacy/LEGACY-M0/manifests/trained_m0_audit.json",
)
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules", "runs", "secrets", "credentials"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".pt", ".pth", ".ckpt", ".safetensors", ".pem", ".key", ".p12", ".bin", ".onnx", ".h5", ".hdf5", ".zip", ".exe", ".dll", ".so"}


def _write_once(path: Path, data: str) -> None:
    encoded = data.encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"Versioned file already exists; create a new revision: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded)


def _cell(kind: str, source: str) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        compile(source, "generated_colab_cell", "exec")
        result.update(execution_count=None, outputs=[])
    return result


def source_commit(root: Path) -> str:
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        if value:
            return value
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        from dulwich.repo import Repo
        with Repo(str(root)) as repository:
            return repository.head().decode("ascii")
    except (ImportError, KeyError) as exc:
        raise RuntimeError("Cannot record Git HEAD; commit project initialization first") from exc


def generate_notebook(root: Path, revision_dir: Path) -> Path:
    root, revision_dir = root.resolve(), revision_dir.resolve()
    relative_revision = revision_dir.relative_to(root).as_posix()
    safe_path(root, relative_revision)
    protocol_path = revision_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_hash = sha256(protocol_path)
    identity = f"{protocol['experiment_id']}/{protocol['revision']}"
    details = {key: protocol.get(key) for key in (
        "hypothesis", "goal_reference", "parent", "command", "inputs", "output_paths",
        "required_artifacts", "gates", "resources", "compute", "prerequisites", "dependencies",
        "scientific_scope", "timeout_seconds",
        "research_track", "paired_question_id", "related_experiments", "research_questions",
    )}
    gate_rows = "\n".join(
        "| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in (
            gate["name"], gate["artifact"] + gate["pointer"], gate["op"], json.dumps(gate["value"]),
        )) + " |" for gate in protocol.get("gates", [])
    )
    heading = f"""# {identity} — {protocol['title']}

این نوت‌بوک فقط برای پروتکل ثبت‌شدهٔ همین نسخه است. هدف، فرضیه، ورودی‌ها،
معیارهای پذیرش و دستور دقیق در زیر ثبت شده‌اند. تغییر روش، seed، داده یا آستانه
نیاز به revision تازه در ریپو دارد. Run all اجرای همین آزمایش را آغاز می‌کند.

**مرحله:** {protocol['stage']} — **نوع:** {protocol['kind']}

**فرضیه:** {protocol['hypothesis']}

**ارجاع هدف:** {protocol['goal_reference']}

```json
{json.dumps(details, ensure_ascii=False, indent=2)}
```

Protocol SHA-256: `{protocol_hash}`

| معیار | فایل و مسیر مقدار | عملگر | مقدار لازم |
|---|---|---|---|
{gate_rows}

این اجرا به‌تنهایی مجوز عبور فاز نیست؛ خروجی باید به ریپو برگردد، با قفل‌های
نسخه تطبیق داده شود، معیارها ارزیابی شوند و نتیجه‌گیری ثبت شود. شکست هم خروجی
پژوهشی است و باید نگه‌داری شود. کنترل داده و checkpoint با SHA-256 انجام می‌شود.

زمان و حافظه تابع تنظیمات پروتکل و runtime است؛ اگر در بخش resources یا compute مقدار ثبت
نشده، برآورد هنوز معلوم نیست. محیط اجرا و مدت واقعی خودکار در نتیجه ثبت می‌شوند.
"""
    preparation = """## آماده‌سازی و دریافت نسخهٔ دقیق

ابتدا runtime مطابق بخش resources یا compute انتخاب شود. دو فایل ساخته‌شده در ریپو را باهم
بارگذاری کنید: ZIP سورس نسخه و `package.json` کنار همین نوت‌بوک. بسته عمداً
checkpointها، نتایج قبلی و محیط مجازی را ندارد. این سلول hash کل بسته و تک‌تک
فایل‌های آن را بررسی می‌کند و در پوشه‌ای تازه باز می‌کند.

هیچ سرویس محلی نمی‌تواند صرف ساخت این نوت‌بوک، اجرای Colab را ادعا کند. اجرای
واقعی با Run all در حساب پژوهشگر یا با ابزار مجاز و متصل انجام می‌شود.
"""
    setup = f'''from pathlib import Path, PurePosixPath
import hashlib, json, os, stat, sys, uuid, zipfile
from google.colab import files

EXPERIMENT_ID = {protocol['experiment_id']!r}
REVISION = {protocol['revision']!r}
PROTOCOL_SHA256 = {protocol_hash!r}
REVISION_RELATIVE = {relative_revision!r}
uploaded = files.upload()
if 'package.json' not in uploaded:
    raise RuntimeError('Upload package.json and its exact source ZIP together')
lock = json.loads(uploaded['package.json'])
assert lock['experiment_id'] == EXPERIMENT_ID and lock['revision'] == REVISION
assert lock['protocol_sha256'] == PROTOCOL_SHA256
archive_name = lock['archive_name']
if archive_name not in uploaded:
    raise RuntimeError('Missing source archive: ' + archive_name)
source_archive = Path(archive_name)
actual_archive_sha = hashlib.sha256(source_archive.read_bytes()).hexdigest()
assert actual_archive_sha == lock['package_sha256'], 'Source ZIP SHA mismatch'
run_id = __import__('datetime').datetime.now(__import__('datetime').timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
ROOT = Path('/content/ergt_runs') / EXPERIMENT_ID / REVISION / run_id / 'source'
ROOT.mkdir(parents=True, exist_ok=False)
with zipfile.ZipFile(source_archive) as archive:
    seen = set()
    total = 0
    for item in archive.infolist():
        name = item.filename
        parts = name.split('/')
        if not name or '\\\\' in name or ':' in name or name.startswith('/') or any(p in ('', '.', '..') for p in parts):
            raise RuntimeError('Unsafe ZIP path: ' + name)
        if name.casefold() in seen or stat.S_ISLNK(item.external_attr >> 16):
            raise RuntimeError('Duplicate or symlink ZIP entry: ' + name)
        seen.add(name.casefold())
        total += item.file_size
        if total > 2 * 1024**3:
            raise RuntimeError('Unexpectedly large source archive')
    manifest_bytes = archive.read('PACKAGE_MANIFEST.json')
    assert hashlib.sha256(manifest_bytes).hexdigest() == lock['package_manifest_sha256']
    package_manifest = json.loads(manifest_bytes)
    assert set(archive.namelist()) == set(package_manifest['files']) | {{'PACKAGE_MANIFEST.json'}}
    archive.extractall(ROOT)
for relative, expected in package_manifest['files'].items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
REVISION_DIR = ROOT / REVISION_RELATIVE
assert hashlib.sha256((REVISION_DIR / 'protocol.json').read_bytes()).hexdigest() == PROTOCOL_SHA256
assert hashlib.sha256((REVISION_DIR / 'experiment.ipynb').read_bytes()).hexdigest() == lock['notebook_sha256']
sys.path.insert(0, str(ROOT))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
protocol = json.loads((REVISION_DIR / 'protocol.json').read_text(encoding='utf-8'))
print('Verified source:', EXPERIMENT_ID, REVISION, actual_archive_sha)
'''
    inputs_text = """## ورودی‌ها و پیش‌نیازها

هر ورودی در `inputs` باید مسیر دقیق داخل پروژه، SHA-256 و محل دسترسی ثبت‌شده
داشته باشد. برای فایل‌های بزرگ از مسیر Drive استفاده کنید؛ در غیر این صورت
سلول برای هر فایل ثبت‌شده پنجرهٔ upload باز می‌کند. فایل‌ها تنها پس از تطبیق
hash کپی می‌شوند. هیچ checkpoint با نام مشابه جایگزین نسخهٔ ثبت‌شده نمی‌شود.

وابستگی تازه یا تغییر محیط باید در پروتکل نسخهٔ بعدی ثبت شود. این نوت‌بوک
به‌صورت پنهان package نصب نمی‌کند. محیط و خروجی خطای وابستگی نیز در bundle
اجرای ناموفق ثبت می‌شود.
"""
    inputs_code = '''from google.colab import drive
drive.mount('/content/drive')
import shutil
from research_tools.runner import safe_path, sha256

INPUT_ERRORS = []
for item in protocol.get('inputs', []):
    try:
        target = safe_path(ROOT, item['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and sha256(target) == item['sha256']:
            continue
        locator = item.get('uri', '')
        candidate = Path(locator.removeprefix('file://')) if locator.startswith(('/content/drive/', 'file:///content/drive/')) else None
        if candidate is not None and candidate.is_file():
            assert sha256(candidate) == item['sha256'], 'Drive input SHA mismatch: ' + item['path']
            shutil.copy2(candidate, target)
        else:
            print('Upload input:', item['path'], 'Registered locator:', locator)
            input_upload = files.upload()
            matched = [name for name, data in input_upload.items() if hashlib.sha256(data).hexdigest() == item['sha256']]
            if len(matched) != 1:
                raise RuntimeError('Exactly one uploaded input must match its registered SHA')
            target.write_bytes(input_upload[matched[0]])
        assert sha256(target) == item['sha256']
    except Exception as exc:
        INPUT_ERRORS.append(item['path'] + ': ' + str(exc))
print('Input errors; runner will export preflight failure evidence:' if INPUT_ERRORS else 'All input locks passed', INPUT_ERRORS)
'''
    execution_text = """## اجرا و حفظ شواهد

Runner دقیقاً command ثبت‌شده را اجرا می‌کند. stdout/stderr، محیط، زمان، کد
خروج و تمام فایل‌های output_paths در ZIP نتیجه ثبت می‌شوند؛ شکست دستور هم
bundle دارد. پوشهٔ نتیجه روی Drive برای هر اجرا مستقل است و بازنویسی نمی‌شود.
در صورت قطع کامل runtime باید اجرای ناتمام ثبت و با run_id تازه تکرار شود.

موفقیت اجرایی (`returncode=0`) به معنی موفقیت علمی یا عبور فاز نیست.
"""
    execution_code = '''from research_tools.runner import run_protocol
DRIVE_RUN = Path('/content/drive/MyDrive/ERGT_Phi/research') / EXPERIMENT_ID / REVISION / run_id
DRIVE_RUN.mkdir(parents=True, exist_ok=False)
RESULT_BUNDLE = DRIVE_RUN / (run_id + '.zip')
run_manifest = run_protocol(ROOT, REVISION_DIR, RESULT_BUNDLE,
                            run_id=run_id, package_sha256=actual_archive_sha)
print(json.dumps(run_manifest, indent=2, ensure_ascii=False))
print('Preserved result bundle:', RESULT_BUNDLE)
'''
    export_text = f"""## برگشت نتیجه و تصمیم بعدی

ZIP را دانلود و در `research/inbox/` ریپو قرار دهید. ایجنت باید آن را با دستور
workflow import دریافت کند، hash و ارتباط پروتکل/نوت‌بوک/سورس را بررسی کند،
و کنار همین revision گزارش تولید کند. سپس نتیجهٔ واقعی، کنترل‌ها، شکست‌ها و
گزینه‌های بعدی در review و دفتر تصمیم معماری ثبت شوند. هر پیشرفت باید commit
شود. اگر فایل نتیجه هنوز برنگشته، وضعیت «در انتظار نتیجه» است.

شناسهٔ پیگیری: `{identity}`. برای تکرار یک نسخه run_id تازه لازم است؛ برای
تغییر فرضیه، کد علمی یا معیار، revision تازه لازم است. این نوت‌بوک را برای
دورزدن معیارها در Colab ویرایش نکنید.
"""
    notebook = {
        "cells": [
            _cell("markdown", heading), _cell("markdown", preparation), _cell("code", setup),
            _cell("markdown", inputs_text), _cell("code", inputs_code),
            _cell("markdown", execution_text), _cell("code", execution_code),
            _cell("markdown", export_text), _cell("code", "files.download(str(RESULT_BUNDLE))\n"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "ergt_phi": {"experiment_id": protocol["experiment_id"], "revision": protocol["revision"], "protocol_sha256": protocol_hash},
        }, "nbformat": 4, "nbformat_minor": 4,
    }
    target = revision_dir / "experiment.ipynb"
    _write_once(target, json.dumps(notebook, ensure_ascii=False, indent=2) + "\n")
    _write_once(revision_dir / "notebook.sha256", sha256(target) + "  experiment.ipynb\n")
    return target


def _allowed(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return not (
        any(part in EXCLUDED_PARTS or part.startswith(".") or part.startswith("pytest-cache-files-") for part in relative.parts)
        or path.suffix.lower() in EXCLUDED_SUFFIXES
        or path.name.lower() in {"credentials.json", "service-account.json", "token.json", "secrets.json"}
    )


def _write_archive(archive: Path, root: Path, manifest_bytes: bytes) -> None:
    manifest = json.loads(manifest_bytes)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as output:
        # Fixed timestamps and inventory order make exact restoration possible.
        members = [("PACKAGE_MANIFEST.json", manifest_bytes)] + [
            (relative, safe_path(root, relative).read_bytes()) for relative in manifest["files"]
        ]
        for relative, data in members:
            info = zipfile.ZipInfo(relative, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            output.writestr(info, data)


def build_source_package(root: Path, revision_dir: Path) -> Path:
    root, revision_dir = root.resolve(), revision_dir.resolve()
    revision_relative = revision_dir.relative_to(root).as_posix()
    safe_path(root, revision_relative)
    protocol_path = revision_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    notebook = revision_dir / "experiment.ipynb"
    if not notebook.is_file():
        raise ValueError("Generate the versioned notebook before packaging")
    lock_path = revision_dir / "package.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        archive = safe_path(root, "research/packages/" + lock["archive_name"])
        if sha256(protocol_path) != lock["protocol_sha256"] or sha256(notebook) != lock["notebook_sha256"]:
            raise ValueError("Registered protocol/notebook changed; create a new revision")
        if archive.is_file():
            if sha256(archive) == lock["package_sha256"]:
                return archive
            raise FileExistsError("Locked source package was modified; restore its archived bytes")
        inventory_path = revision_dir / "source_manifest.json"
        if not inventory_path.is_file() or sha256(inventory_path) != lock["package_manifest_sha256"]:
            raise ValueError("Missing or modified source_manifest.json; restore the tracked release inventory")
        inventory_bytes = inventory_path.read_bytes()
        inventory = json.loads(inventory_bytes)
        for relative, expected in inventory["files"].items():
            current = safe_path(root, relative)
            if not current.is_file() or sha256(current) != expected:
                raise ValueError(
                    f"Cannot restore archived source from changed file: {relative}. "
                    f"Restore the recorded files from Git commit {lock['source_commit']} "
                    "in a separate checkout, then rebuild this revision."
                )
        _write_archive(archive, root, inventory_bytes)
        if sha256(archive) != lock["package_sha256"]:
            raise ValueError("Restored ZIP differs from locked bytes; preserve it for diagnosis and restore original archive")
        return archive
    files = []
    for name in SOURCE_DIRECTORIES:
        base = root / name
        if base.exists():
            for path in base.rglob("*"):
                if path.is_file() and _allowed(path, root):
                    safe_path(root, path.relative_to(root).as_posix())
                    files.append(path)
    for name in SOURCE_FILES:
        if (root / name).is_file():
            files.append(safe_path(root, name))
    files.extend((protocol_path, notebook, revision_dir / "notebook.sha256"))
    files = sorted(set(files))
    inventory = {path.relative_to(root).as_posix(): sha256(path) for path in files}
    commit = source_commit(root)
    package_manifest = {
        "schema": "ergt-phi-source-v1", "experiment_id": protocol["experiment_id"],
        "revision": protocol["revision"], "source_commit": commit,
        "protocol_sha256": sha256(protocol_path), "notebook_sha256": sha256(notebook),
        "files": inventory,
        "source_policy": "Exact file hashes identify working-tree contents; source_commit records the base Git HEAD",
    }
    manifest_bytes = (json.dumps(package_manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    _write_once(revision_dir / "source_manifest.json", manifest_bytes.decode("utf-8"))
    archive_name = f"{protocol['experiment_id']}-{protocol['revision']}-{manifest_hash[:12]}-source.zip"
    archive = safe_path(root, "research/packages/" + archive_name)
    _write_archive(archive, root, manifest_bytes)
    lock = {
        "schema": "ergt-phi-package-lock-v1", "experiment_id": protocol["experiment_id"],
        "revision": protocol["revision"], "archive_name": archive_name,
        "package_sha256": sha256(archive), "package_manifest_sha256": manifest_hash,
        "protocol_sha256": package_manifest["protocol_sha256"],
        "notebook_sha256": package_manifest["notebook_sha256"], "source_commit": commit,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(inventory), "bytes": archive.stat().st_size,
    }
    _write_once(lock_path, json.dumps(lock, ensure_ascii=False, indent=2) + "\n")
    return archive

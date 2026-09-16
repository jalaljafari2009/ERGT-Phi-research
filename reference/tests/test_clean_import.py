import os
import subprocess
import sys
import tempfile
from pathlib import Path


def test_clean_import_outside_release():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as outside:
        code = (
            "import pathlib,sys;"
            f"root=pathlib.Path({str(root)!r}).resolve();"
            "sys.path.insert(0,str(root));"
            "import ergt_four_seed;"
            "from ergt_four_seed.runtime import activate_locked_runtime;"
            "activate_locked_runtime();"
            "import ergt_reviewer.v9_confirmation as v9;"
            "assert root in pathlib.Path(ergt_four_seed.__file__).resolve().parents;"
            "assert root in pathlib.Path(v9.__file__).resolve().parents;"
            "print(v9.SCHEMA_VERSION)"
        )
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment["PYTHONNOUSERSITE"] = "1"
        completed = subprocess.run(
            [sys.executable, "-c", code], cwd=outside, env=environment,
            check=True, text=True, capture_output=True,
        )
    assert "ergt-geometric-long-horizon-study-v1" in completed.stdout

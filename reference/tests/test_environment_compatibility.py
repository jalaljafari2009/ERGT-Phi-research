import json
from pathlib import Path

from ergt_four_seed.environment import verify_environment


def test_compatible_runtime_is_not_blocked_by_reference_version_differences():
    environment = verify_environment(strict=True)
    assert environment["pass"] is True
    assert environment["reference_match_required"] is False
    assert all(environment["checks"].values())
    assert environment["notebook_packages_installed_if_missing"]["pandas"] == "pandas"


def test_notebook_names_the_exact_upload_and_automatic_dependency_policy():
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / "notebook/ERGT_Attention_Free_Geometric_Reasoning_Study.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join("".join(cell.get("source", ())) for cell in notebook["cells"])
    assert "ERGT_FourSeed_Reproduction.zip" in source
    assert "ERGT-paper-main/ERGT_FourSeed_Reproduction.zip" in source
    assert "Code > Download ZIP" in source
    assert "extract the downloaded `ERGT-paper-main.zip`" in source
    assert "do not upload ergt-paper-main.zip" in source.lower()
    assert "ensure_required_packages(install_missing=True)" in source
    assert "no historical colab runtime version is required" in source.lower()

"""Run the local research workflow without loading the model or PyTorch."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


if __name__ == "__main__":
    from research_tools.workflow import main

    raise SystemExit(main(root=ROOT))

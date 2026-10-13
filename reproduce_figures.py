"""Reproduce the manuscript figures from supplied numerical data and photos."""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code/figure_exports"))

from composite_figures import main


if __name__ == "__main__":
    main()
    if "--only" not in sys.argv:
        subprocess.run([sys.executable, str(ROOT / "code/figure_exports/moxi_monitoring.py")], check=True)

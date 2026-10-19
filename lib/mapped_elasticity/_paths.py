"""Local paths for the distributed solver modules."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parent
OUT = PROJECT / "results/data/generated"
RESULTS = RESULTS_FINAL = OUT

def bridge():
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))

bridge()

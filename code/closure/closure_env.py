"""Locate the solver modules included in this research archive."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def bridge_pinned():
    library = PROJECT_ROOT / "lib"
    if str(library) not in sys.path:
        sys.path.insert(0, str(library))
    return {"solver_root": str(library)}

from __future__ import annotations

import sys
from pathlib import Path

# Add the repo root so that `scripts.*` modules are importable in tests.
_root = str(Path(__file__).parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

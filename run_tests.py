#!/usr/bin/env python
"""Offline regression runner for Jev-GamePilot.

Usage:
    python run_tests.py            # full offline suite
    python run_tests.py -v         # verbose
    python run_tests.py tests.test_clash_ladder   # one module

Sets JEV_OFFLINE=1 before discovery so no model weights load and no network
tier is queried. The suite mocks desktop APIs: it never sends input events.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(argv: list[str]) -> int:
    os.environ["JEV_OFFLINE"] = "1"
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    verbosity = 2 if "-v" in argv else 1
    targets = [a for a in argv if not a.startswith("-")]

    if targets:
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromNames(targets)
    else:
        suite = unittest.TestLoader().discover(
            start_dir=str(ROOT / "tests"), top_level_dir=str(ROOT)
        )
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

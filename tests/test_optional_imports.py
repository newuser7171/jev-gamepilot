"""Optional backends must never become import-time requirements.

The cloud SDK (`typesafe_sdk`) and the chess engine (`chess`) are installed only
when a user enables those tiers. Every module that touches them is imported
eagerly by the pilot, so a heuristic-only install must still import the whole
stack. Runs each import in a child interpreter with the packages hidden: the
same conditions as the CI runner and a fresh `pip install -r requirements.txt`.
"""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BLOCKED = ("chess", "typesafe_sdk")

MODULES = (
    "universal_brain",
    "jev_brain",
    "pilot_core",
    "embedded_dino",
    "adapters.clash_adapter",
    "adapters.screen_chess_adapter",
)

_CHILD = f"""
import importlib
import importlib.abc
import sys

BLOCKED = {BLOCKED!r}


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError("No module named %r" % fullname, name=fullname)
        return None


sys.meta_path.insert(0, _Blocker())

for name in {MODULES!r}:
    importlib.import_module(name)
print("ok")
"""


class OptionalBackendImportTests(unittest.TestCase):
    def test_stack_imports_without_optional_backends(self):
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr.strip()[-2000:])
        self.assertIn("ok", proc.stdout)


if __name__ == "__main__":
    unittest.main()

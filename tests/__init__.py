"""Offline regression suite.

Importing this package forces ``JEV_OFFLINE=1`` before any test module pulls in
``universal_brain``: no weight loaders, no model downloads, no network tiers.
Run with ``python run_tests.py`` (which uses ``-t .`` so this guard executes).
"""

import os

os.environ.setdefault("JEV_OFFLINE", "1")

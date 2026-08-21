"""Load modeling framework tests without ``tests/modeling`` shadowing ``src/modeling``."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str) -> unittest.TestSuite:
    root = Path(__file__).resolve().parent.parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    for key in list(sys.modules):
        if key == "modeling" or key.startswith("modeling."):
            del sys.modules[key]

    suite = unittest.TestSuite()
    modeling_dir = Path(__file__).resolve().parent / "modeling"
    glob_pattern = pattern if pattern and pattern != "test_*.py" else "framework_test_*.py"
    for path in sorted(modeling_dir.glob(glob_pattern)):
        module_name = f"gas_wait_modeling_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        suite.addTests(loader.loadTestsFromModule(module))
    return suite

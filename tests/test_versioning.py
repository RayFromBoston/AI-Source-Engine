import pathlib
import re
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from al10 import __version__


class TestVersioning(unittest.TestCase):
    def _read_pyproject_version(self) -> str:
        pyproject_path = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
        text = pyproject_path.read_text(encoding="utf-8")
        # Python 3.10-compatible lightweight parse for: version = "x.y.z"
        match = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"\s*$', text)
        if not match:
            self.fail("could not find version in pyproject.toml")
        return match.group(1)

    def test_pyproject_version_matches_package(self) -> None:
        self.assertEqual(self._read_pyproject_version(), __version__)

if __name__ == "__main__":
    unittest.main()

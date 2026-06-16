import pathlib
import sys
import tomllib
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from al10 import __version__


class TestVersioning(unittest.TestCase):
    def test_pyproject_version_matches_package(self) -> None:
        pyproject_path = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        self.assertEqual(pyproject["project"]["version"], __version__)

    def test_changelog_has_current_version(self) -> None:
        changelog_path = pathlib.Path(__file__).resolve().parents[1] / "CHANGELOG.md"
        changelog = changelog_path.read_text(encoding="utf-8")
        self.assertIn(f"## [{__version__}]", changelog)


if __name__ == "__main__":
    unittest.main()

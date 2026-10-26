"""Structural checks: no scientific dependencies, network calls or model training."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_paths import raw_corpus_paths


class ProjectLayoutTests(unittest.TestCase):
    def test_images_are_unique(self):
        images = list((ROOT / "assets/images").iterdir())
        self.assertTrue(images)
        digests = {}
        for image in images:
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            self.assertNotIn(digest, digests, f"Duplicate image: {image.name}")
            digests[digest] = image.name

    def test_app_opens_real_resources_from_another_working_directory(self):
        # Replace only external libraries. Execute the real entry point until its
        # first CSV load, without installing the legacy stack or training a model.
        smoke = '''
import csv, json, runpy, sys
from pathlib import Path
from unittest.mock import MagicMock

for name in ["streamlit", "pandas", "numpy", "scipy", "scipy.sparse", "PIL",
             "matplotlib", "matplotlib.pyplot", "seaborn", "bokeh", "bokeh.io",
             "bokeh.palettes", "bokeh.plotting", "bokeh.transform", "bokeh.models",
             "bokeh.models.widgets", "bokeh.layouts"]:
    sys.modules[name] = MagicMock()
def passthrough(*args, **kwargs):
    return lambda function: function
sys.modules["streamlit"].cache_data = passthrough
sys.modules["streamlit"].cache_resource = passthrough
sys.modules["streamlit"].tabs.return_value = [MagicMock(), MagicMock(), MagicMock()]
seen = []
class DatasetReached(Exception):
    pass
def open_image(path):
    path = Path(path)
    assert path.is_absolute() and path.is_file(), path
    seen.append(str(path))
    return MagicMock()
def read_csv(path, **kwargs):
    path = Path(path)
    assert path.is_absolute() and path.is_file(), path
    with path.open(newline="") as stream:
        assert "target" in next(csv.reader(stream))
    seen.append(str(path))
    raise DatasetReached()
sys.modules["PIL"].Image.open.side_effect = open_image
sys.modules["pandas"].read_csv.side_effect = read_csv
try:
    runpy.run_path(sys.argv[1], run_name="__layout_smoke__")
except DatasetReached:
    print(json.dumps(seen))
else:
    raise AssertionError("App did not load its dataset")
'''
        with TemporaryDirectory() as folder:
            env = {**os.environ, "PYTHONPATH": str(ROOT)}
            result = subprocess.run(
                [sys.executable, "-c", smoke, str(ROOT / "app.py")],
                cwd=folder, env=env, capture_output=True, text=True, check=True,
            )
        paths = [Path(path) for path in json.loads(result.stdout)]
        self.assertEqual(paths[0].parent, ROOT / "assets/images")
        self.assertEqual(paths[-1], ROOT / "data/processed/spam_ham.csv")

    def test_notebook_configuration_uses_the_same_external_corpus(self):
        notebook = json.loads((ROOT / "notebooks/spam_detector.ipynb").read_text())
        self.assertEqual(notebook["nbformat"], 4)
        setups = ["".join(cell.get("source", [])) for cell in notebook["cells"]
                  if "raw_corpus_paths()" in "".join(cell.get("source", []))]
        self.assertEqual(len(setups), 1)
        previous_cwd = Path.cwd()
        previous_sys_path = list(sys.path)
        try:
            with TemporaryDirectory() as folder:
                corpus = Path(folder)
                for name in ("spam", "ham"):
                    (corpus / name).mkdir()
                with patch.dict(os.environ, {"SPAM_DETECTOR_RAW_DATA": str(corpus)}):
                    for cwd in (ROOT, ROOT / "notebooks"):
                        os.chdir(cwd)
                        namespace = {}
                        exec(compile(setups[0], "notebook-configuration", "exec"), namespace)
                        self.assertEqual(namespace["spam_path"], corpus / "spam")
                        self.assertEqual(namespace["ham_path"], corpus / "ham")
        finally:
            os.chdir(previous_cwd)
            sys.path[:] = previous_sys_path

    def test_missing_corpus_reports_the_required_configuration(self):
        with TemporaryDirectory() as folder:
            (Path(folder) / "spam").mkdir()
            with patch.dict(os.environ, {"SPAM_DETECTOR_RAW_DATA": folder}):
                with self.assertRaisesRegex(FileNotFoundError, "SPAM_DETECTOR_RAW_DATA"):
                    raw_corpus_paths()

    def test_relative_corpus_path_is_rejected(self):
        with patch.dict(os.environ, {"SPAM_DETECTOR_RAW_DATA": "relative/corpus"}):
            with self.assertRaisesRegex(ValueError, "absolute"):
                raw_corpus_paths()

    def test_readme_local_links_resolve(self):
        for link in re.findall(r"\]\(([^)]+)\)", (ROOT / "README.md").read_text()):
            if not link.startswith(("http://", "https://", "#")):
                self.assertTrue((ROOT / link.split("#")[0]).exists(), link)


if __name__ == "__main__":
    unittest.main()

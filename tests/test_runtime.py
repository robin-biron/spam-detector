"""Exercise the actual article and models when application dependencies are installed."""

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
HAS_RUNTIME = all(importlib.util.find_spec(name) for name in ("streamlit", "sklearn", "pandas"))


@unittest.skipUnless(HAS_RUNTIME, "Install requirements.txt to test the deployed application")
class ApplicationRuntimeTests(unittest.TestCase):
    def test_article_models_resources_and_cached_rerun(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
        self.assertEqual(len(app.exception), 0, [error.message for error in app.exception])
        self.assertIn("Spam Detector", app.title[0].value)
        scores = [(metric.label, metric.value) for metric in app.metric]
        self.assertEqual(len(scores), 3)
        for _, score in scores:
            self.assertGreaterEqual(float(score.rstrip("%")), 0)
            self.assertLessEqual(float(score.rstrip("%")), 100)
        self.assertNotIn("Unnamed: 0", app.dataframe[0].value.columns)
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(scores, [(metric.label, metric.value) for metric in app.metric])


if __name__ == "__main__":
    unittest.main()

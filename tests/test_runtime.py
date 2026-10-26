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
        self.assertEqual([tab.label for tab in app.tabs], [
            "Spam & ham examples", "Analyze an email", "Methodology & results",
        ])
        for category in ("Spam", "Ham"):
            app.radio[0].set_value(category).run()
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("Prediction:" in element.value
                                for element in [*app.warning, *app.success]))
            probability = next(metric for metric in app.metric
                               if metric.label == "Estimated spam probability")
            self.assertGreaterEqual(float(probability.value.rstrip("%")), 0)
            self.assertLessEqual(float(probability.value.rstrip("%")), 100)
        app.text_area[1].input("   ")
        app.button[1].click().run()
        self.assertIn("Enter an email", app.error[0].value)
        app.text_area[1].input("zzqvxkplmzz zzqvxkplmzz")
        app.button[1].click().run()
        self.assertIn("No words match", app.error[0].value)
        app.text_area[1].input("Hi team, please review the project report before our meeting tomorrow. Thank you.")
        app.button[1].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(sum(metric.label == "Estimated spam probability"
                             for metric in app.metric), 1)
        self.assertEqual(scores, [(metric.label, metric.value) for metric in app.metric if metric.label != "Estimated spam probability"])


if __name__ == "__main__":
    unittest.main()

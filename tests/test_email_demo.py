"""Check rejection before model access, including the server-side size limit."""

import unittest

from email_demo import MAX_EMAIL_LENGTH, classify_email


class EmailValidationTests(unittest.TestCase):
    def test_empty_email_is_rejected_before_model_access(self):
        with self.assertRaisesRegex(ValueError, "Enter an email"):
            classify_email(" \n\t", None, None)

    def test_oversized_email_is_rejected_before_model_access(self):
        with self.assertRaisesRegex(ValueError, "20,000"):
            classify_email("a" * (MAX_EMAIL_LENGTH + 1), None, None)


if __name__ == "__main__":
    unittest.main()

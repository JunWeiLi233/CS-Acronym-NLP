"""Regression checks for keeping training and evaluation data separate."""

import unittest

from utils import reject_evaluation_overlap


class HeldOutSeparationTests(unittest.TestCase):
    def test_rejects_reused_document_under_different_sentence_id(self):
        train = [{"id": "train-a", "doc_id": "paper", "tokens": ["First"]}]
        test = [{"id": "test-b", "doc_id": "paper", "tokens": ["Later"]}]

        with self.assertRaisesRegex(ValueError, "overlaps the evaluation IDs"):
            reject_evaluation_overlap(train, test, "train.jsonl")

    def test_rejects_duplicate_text_even_under_new_record_id_and_casing(self):
        train = [{"id": "train-a", "tokens": ["Language", "Models"]}]
        test = [{"id": "test-b", "tokens": ["language", "models"]}]

        with self.assertRaisesRegex(ValueError, "repeats an evaluation sentence"):
            reject_evaluation_overlap(train, test, "train.jsonl")

    def test_accepts_distinct_documents_and_sentences(self):
        train = [{"id": "train-a", "doc_id": "a", "tokens": ["First"]}]
        test = [{"id": "test-b", "doc_id": "b", "tokens": ["Later"]}]

        reject_evaluation_overlap(train, test, "train.jsonl")


if __name__ == "__main__":
    unittest.main()

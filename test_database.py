import unittest
import sqlite3
from database import DatabaseManager

class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        # Use an in-memory database for testing
        self.db = DatabaseManager(db_path="file::memory:?cache=shared")

    def test_save_vocab_inserts_data(self):
        # Happy path: inserting a simple vocabulary
        vocab = {'apple': 0, 'banana': 1, 'cherry': 2}
        self.db.save_vocab(vocab)

        # Load the vocabulary to verify
        loaded_vocab = self.db.load_vocab()
        self.assertEqual(vocab, loaded_vocab)

    def test_save_vocab_overwrites_existing(self):
        # Overwrite path: save one vocabulary, then another
        vocab1 = {'apple': 0, 'banana': 1}
        self.db.save_vocab(vocab1)

        vocab2 = {'dog': 0, 'cat': 1, 'mouse': 2}
        self.db.save_vocab(vocab2)

        # Verify that the loaded vocabulary exactly matches the second one
        loaded_vocab = self.db.load_vocab()
        self.assertEqual(vocab2, loaded_vocab)
        # Ensure the old data is completely gone
        self.assertNotIn('apple', loaded_vocab)
        self.assertNotIn('banana', loaded_vocab)

    def test_save_vocab_empty(self):
        # Empty vocabulary
        vocab = {'apple': 0}
        self.db.save_vocab(vocab)

        empty_vocab = {}
        self.db.save_vocab(empty_vocab)

        loaded_vocab = self.db.load_vocab()
        self.assertEqual(empty_vocab, loaded_vocab)
        self.assertEqual(len(loaded_vocab), 0)

if __name__ == '__main__':
    unittest.main()

import unittest
import tempfile
import os
from database import DatabaseManager

class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        # Use a temporary file for testing to mimic actual DB behavior
        self.fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.db = DatabaseManager(self.db_path)

    def tearDown(self):
        # Clean up the temporary database file
        os.close(self.fd)
        os.remove(self.db_path)

    def test_save_and_load_vocab(self):
        """Test that a vocabulary can be saved and loaded correctly."""
        vocab = {"hello": 0, "world": 1, "quantum": 2}
        self.db.save_vocab(vocab)
        loaded_vocab = self.db.load_vocab()
        self.assertEqual(loaded_vocab, vocab)

    def test_save_vocab_overwrites_existing(self):
        """Test that saving a new vocabulary completely overwrites the existing one."""
        initial_vocab = {"old": 0, "words": 1}
        self.db.save_vocab(initial_vocab)

        new_vocab = {"new": 0, "vocabulary": 1, "items": 2}
        self.db.save_vocab(new_vocab)

        loaded_vocab = self.db.load_vocab()
        self.assertEqual(loaded_vocab, new_vocab)
        self.assertNotIn("old", loaded_vocab)

    def test_save_empty_vocab(self):
        """Test that saving an empty vocabulary clears the existing one."""
        initial_vocab = {"word": 0}
        self.db.save_vocab(initial_vocab)

        self.db.save_vocab({})
        loaded_vocab = self.db.load_vocab()
        self.assertEqual(loaded_vocab, {})

if __name__ == "__main__":
    unittest.main()

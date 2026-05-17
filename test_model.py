import unittest
import torch
from model import QuantumTransformer

class TestQuantumTransformer(unittest.TestCase):
    def test_count_parameters(self):
        # Instantiate with small parameters for fast testing
        model = QuantumTransformer(
            vocab_size=100,
            embed_dim=16,
            max_seq_len=32,
            num_classes=2,
            quantum_stride=8
        )

        # Test that count_parameters returns an integer > 0
        initial_count = model.count_parameters()
        self.assertIsInstance(initial_count, int)
        self.assertGreater(initial_count, 0)

        # Manually calculate expected parameters
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertEqual(initial_count, manual_count)

        # Test that freezing parameters decreases the count
        # Freeze the embedding layer
        for p in model.embedding.parameters():
            p.requires_grad = False

        frozen_count = model.count_parameters()
        self.assertLess(frozen_count, initial_count)

        # Unfreeze
        for p in model.embedding.parameters():
            p.requires_grad = True

if __name__ == '__main__':
    unittest.main()

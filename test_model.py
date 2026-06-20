import pytest
import torch
from model import QuantumTransformer, VOCAB_SIZE, NUM_CLASSES, MAX_SEQ_LEN

@pytest.mark.parametrize("batch_size, seq_len", [
    (1, 16),
    (4, MAX_SEQ_LEN),
    (2, 32),
])
def test_quantum_transformer_forward(batch_size, seq_len):
    # Initialize model with default arguments
    model = QuantumTransformer()

    # Set to eval mode for deterministic testing if dropout/batchnorm are used
    model.eval()

    # Create a dummy input tensor
    # Ensure inputs are valid token IDs (0 to VOCAB_SIZE - 1)
    dummy_input = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len))

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Assert expected shape
    assert output.shape == (batch_size, NUM_CLASSES), \
        f"Expected output shape {(batch_size, NUM_CLASSES)}, but got {output.shape}"

    # Assert output contains no NaNs
    assert not torch.isnan(output).any(), "Model output contains NaNs"

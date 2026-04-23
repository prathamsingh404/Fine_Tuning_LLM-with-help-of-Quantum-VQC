"""
quantum_layer.py
================
Defines the Variational Quantum Circuit (VQC) and the QuantumAttentionLayer
with advanced architecture: Data Re-uploading, Strongly Entangling Layers, 
and Residual Connections.
"""

import math
import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────────────────────────────────────
N_QUBITS = 10       # Number of qubits
N_LAYERS = 3        # Number of variational layers (increased from 1)
ENTANGLEMENT = "all-to-all" # Type of entanglement

# Create PennyLane device
# Using 'default.qubit' for exact simulation
dev = qml.device("default.qubit", wires=N_QUBITS)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Quantum Circuit Definition with Data Re-uploading
# ─────────────────────────────────────────────────────────────────────────────
@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    """
    Advanced VQC with Data Re-uploading and Strong Entanglement.
    
    Structure:
        For each layer:
            1. Data Re-uploading (RY/RZ encoding of classical inputs)
            2. Strongly Entangling Layers (Trainable RY+RZ+CNOTs)
        Final Measurement: PauliZ expectation values.
        
    Args:
        inputs (torch.Tensor): Shape (N_QUBITS*2,) -- 2 values per qubit to encode
        weights (torch.Tensor): Shape (N_LAYERS, N_QUBITS, 3) -- RZ, RY, RZ
    """
    # Reshape inputs to (N_QUBITS, 2) for RY/RZ encoding
    # We encode 20 inputs into 10 qubits (one RY, one RZ per qubit)
    enc_inputs = inputs.reshape(N_QUBITS, 2)

    for layer in range(N_LAYERS):
        # --- 1. Data Re-uploading Layer ---
        for i in range(N_QUBITS):
            qml.RY(enc_inputs[i, 0] * math.pi, wires=i)
            qml.RZ(enc_inputs[i, 1] * math.pi, wires=i)

        # --- 2. Trainable Variational Layer (Strong Entanglement) ---
        # We manually implement a strongly entangling-like structure for better control
        # RY + RZ trainable rotations
        for i in range(N_QUBITS):
            qml.RZ(weights[layer, i, 0], wires=i)
            qml.RY(weights[layer, i, 1], wires=i)
            qml.RZ(weights[layer, i, 2], wires=i)

        # Entanglement structure: circular CNOTs (can be expanded to all-to-all)
        # Circular CNOT is a good hardware-efficient default.
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])

    # Measurement: Pauli-Z expectation values
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# ─────────────────────────────────────────────────────────────────────────────
# 3. QuantumAttentionLayer
# ─────────────────────────────────────────────────────────────────────────────
class QuantumAttentionLayer(nn.Module):
    """
    Hybrid Classical-Quantum Attention Layer with Data Re-uploading.
    
    Features:
        - 20-dim bottleneck mapped to 10 qubits (2 features per qubit)
        - Data re-uploading across multiple variational layers
        - Optional Residual Connection
    """

    def __init__(self, input_dim: int, stride: int = 8, residual: bool = True):
        super(QuantumAttentionLayer, self).__init__()
        
        self.input_dim = input_dim
        self.stride    = stride
        self.residual  = residual
        
        # Mapping D -> 20 (2 features per qubit for 10 qubits)
        self.bottleneck_dim = N_QUBITS * 2
        self.input_proj = nn.Linear(input_dim, self.bottleneck_dim)
        
        # Variational Weights: 3 angles (RZ, RY, RZ) per qubit per layer
        weight_shapes = {"weights": (N_LAYERS, N_QUBITS, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        # Scaling layer to initialize near identity (Avoid Barren Plateau)
        self.q_scale = nn.Parameter(torch.ones(1) * 0.1) # Start small
        
        # Output projection: 10 qubits -> D
        self.output_proj = nn.Linear(N_QUBITS, input_dim)
        self.tanh = nn.Tanh()

        # Identity initialization for weights to avoid barren plateau
        nn.init.uniform_(self.qlayer.weights, -0.01, 0.01)

    def _run_vqc_on_batch(self, x_flat: torch.Tensor) -> torch.Tensor:
        """Batch execution of VQC."""
        outputs = []
        for i in range(x_flat.shape[0]):
            out = self.qlayer(x_flat[i])
            outputs.append(out)
        return torch.stack(outputs, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Residual Connection.
        x_out = x + Quantum_Projected(x)
        """
        identity = x
        
        # Handle sequence or flat input
        if x.dim() == 2:
            B, D = x.shape
            x_proj = self.tanh(self.input_proj(x))
            x_q = self._run_vqc_on_batch(x_proj) * self.q_scale
            out = self.output_proj(x_q)
        else:
            B, S, D = x.shape
            x_proj = self.tanh(self.input_proj(x)) # (B, S, 20)
            
            # Subsampling for efficiency
            indices = list(range(0, S, self.stride))
            x_sub   = x_proj[:, indices, :]
            n_sub   = x_sub.shape[1]
            
            x_flat  = x_sub.reshape(B * n_sub, self.bottleneck_dim)
            x_q_flat = self._run_vqc_on_batch(x_flat) * self.q_scale
            x_q_sub = x_q_flat.reshape(B, n_sub, N_QUBITS)
            
            # Interpolate back
            x_q_full = torch.nn.functional.interpolate(
                x_q_sub.permute(0, 2, 1),
                size=S,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)
            
            out = self.output_proj(x_q_full)

        # Residual Connection (Crucial for deep transformers)
        if self.residual:
            return out + identity
        return out

if __name__ == "__main__":
    print("-" * 30)
    print("Testing QuantumAttentionLayer (Data Re-uploading)")
    print("-" * 30)
    layer = QuantumAttentionLayer(input_dim=64, stride=16)
    dummy = torch.randn(1, 32, 64)
    out = layer(dummy)
    print(f"Input: {dummy.shape} -> Output: {out.shape}")
    assert out.shape == dummy.shape
    print("Self-test PASSED")

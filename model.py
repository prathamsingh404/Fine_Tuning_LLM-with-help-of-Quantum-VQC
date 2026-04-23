"""
model.py
========
Defines high-performance Quantum and Classical Transformers for Sentiment Analysis.
Both models now feature Residual Connections and optimized bottlenecks.
"""

import math
import torch
import torch.nn as nn
from quantum_layer import QuantumAttentionLayer, N_QUBITS

# ─────────────────────────────────────────────────────────────────────────────
# Shared hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
VOCAB_SIZE    = 10_000   
EMBED_DIM     = 64       
MAX_SEQ_LEN   = 128      
N_HEADS       = 4        
DIM_FF        = 128      
DROPOUT       = 0.1      
NUM_CLASSES   = 2        

# Architecture Constants
BOTTLENECK_DIM = 20 # Increased from 10 to improve information flow

# ─────────────────────────────────────────────────────────────────────────────
# Model 1: QuantumTransformer
# ─────────────────────────────────────────────────────────────────────────────
class QuantumTransformer(nn.Module):
    """
    Quantum-Enhanced Transformer with Residual VQC Block.
    """

    def __init__(
        self,
        vocab_size:     int = VOCAB_SIZE,
        embed_dim:      int = EMBED_DIM,
        max_seq_len:    int = MAX_SEQ_LEN,
        num_classes:    int = NUM_CLASSES,
        quantum_stride: int = 8,
    ):
        super(QuantumTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.zeros(max_seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

        self.encoder1 = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=N_HEADS,
            dim_feedforward=DIM_FF,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.encoder2 = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=N_HEADS,
            dim_feedforward=DIM_FF,
            dropout=DROPOUT,
            batch_first=True,
        )

        # Quantum Layer with Residual=True
        self.quantum_attn = QuantumAttentionLayer(
            embed_dim, 
            stride=quantum_stride, 
            residual=True
        )

        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(DROPOUT)

        print(f"\n[QuantumTransformer] Initialized (N_LAYERS=3, Re-uploading, Residual)")
        print(f"  Params: {self.count_parameters():,}")

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(x)
        seq_len = x.shape[1]
        embeds = embeds + self.pos_encoding[:seq_len]
        embeds = self.dropout(embeds)

        enc1 = self.encoder1(embeds)
        enc2 = self.encoder2(enc1)

        # Non-linear quantum enhancement
        q_out = self.quantum_attn(enc2)

        pooled = q_out.mean(dim=1) 
        logits = self.classifier(pooled)
        return logits

# ─────────────────────────────────────────────────────────────────────────────
# Model 2: ClassicalTransformer (Optimized Baseline)
# ─────────────────────────────────────────────────────────────────────────────
class ClassicalTransformer(nn.Module):
    """
    Classical Baseline with identical Residual structure for fair comparison.
    """

    def __init__(
        self,
        vocab_size:  int = VOCAB_SIZE,
        embed_dim:   int = EMBED_DIM,
        max_seq_len: int = MAX_SEQ_LEN,
        num_classes: int = NUM_CLASSES,
        residual: bool = True
    ):
        super(ClassicalTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = nn.Parameter(torch.zeros(max_seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

        self.encoder1 = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=N_HEADS, dim_feedforward=DIM_FF, 
            dropout=DROPOUT, batch_first=True,
        )
        self.encoder2 = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=N_HEADS, dim_feedforward=DIM_FF, 
            dropout=DROPOUT, batch_first=True,
        )

        self.residual = residual
        # Mirror the quantum block's shape: D -> 20 -> D.
        self.classical_attn = nn.Sequential(
            nn.Linear(embed_dim, BOTTLENECK_DIM),
            nn.Tanh(),
            nn.Linear(BOTTLENECK_DIM, embed_dim),
        )

        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(DROPOUT)

        print(f"\n[ClassicalTransformer] Initialized (Bottleneck={BOTTLENECK_DIM}, Residual={residual})")
        print(f"  Params: {self.count_parameters():,}")

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(x)
        seq_len = x.shape[1]
        embeds = embeds + self.pos_encoding[:seq_len]
        embeds = self.dropout(embeds)

        enc1 = self.encoder1(embeds)
        enc2 = self.encoder2(enc1)

        c_out = self.classical_attn(enc2)
        
        # Fair residual connection
        if self.residual:
            c_out = c_out + enc2

        pooled = c_out.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

if __name__ == "__main__":
    dummy_tokens = torch.randint(0, VOCAB_SIZE, (1, 16))
    q_model = QuantumTransformer()
    c_model = ClassicalTransformer()
    print(f"Q-Out: {q_model(dummy_tokens).shape}")
    print(f"C-Out: {c_model(dummy_tokens).shape}")

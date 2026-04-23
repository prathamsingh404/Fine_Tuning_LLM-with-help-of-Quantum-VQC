"""
train.py
========
High-performance training pipeline with AdamW, Cosine Annealing, 
Label Smoothing, and SQLite logging.
"""

import json
import time
import os
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

# ── Local ─────────────────────────────────────────────────────────────────────
from model import QuantumTransformer, ClassicalTransformer
from database import DatabaseManager

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
VOCAB_SIZE           = 10_000
MAX_LEN              = 128
BATCH_TRAIN          = 32
BATCH_VAL            = 64
EPOCHS               = 25        # Increased for better convergence
QUANTUM_EPOCHS       = 10        # Increased (VQC is slower but needs epochs)
QUANTUM_STRIDE       = 16        # Improved resolution (16 tokens instead of 4)
QUANTUM_TRAIN_SUBSET = 2000      # Reduced subset for faster development
LR                   = 5e-4      # Lower LR for AdamW stability
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_SMOOTHING      = 0.1       # To prevent over-confidence

# Initialize DB
db = DatabaseManager()

# ─────────────────────────────────────────────────────────────────────────────
# 1. Processing Logic
# ─────────────────────────────────────────────────────────────────────────────
def tokenize(sentence, vocab, max_len=MAX_LEN):
    words = str(sentence).lower().split()
    tokens = [vocab.get(w, 1) for w in words]
    tokens = tokens[:max_len]
    tokens += [0] * (max_len - len(tokens))
    return tokens

class SSTDataset(Dataset):
    def __init__(self, sentences, labels, vocab, max_len=MAX_LEN):
        self.labels = labels
        self.tokens = [tokenize(s, vocab, max_len) for s in tqdm(sentences, desc="Tokenizing")]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return torch.tensor(self.tokens[idx], dtype=torch.long), \
               torch.tensor(self.labels[idx], dtype=torch.long)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Advanced Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, train_loader, val_loader, epochs, model_name):
    model = model.to(DEVICE)
    exp_id = db.log_experiment(
        model_name=model_name,
        params=model.count_parameters(),
        best_acc=0.0,
        total_time=0.0,
        config={"lr": LR, "epochs": epochs, "optimizer": "AdamW", "scheduler": "Cosine"}
    )

    # 1. Faster Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 2. Label Smoothing Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_corr, t_total = 0, 0, 0
        
        train_bar = tqdm(train_loader, desc=f"{model_name} [E{epoch}]", leave=False)
        for tokens, labels in train_bar:
            tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(tokens)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            t_loss += loss.item() * tokens.size(0)
            t_corr += (logits.argmax(1) == labels).sum().item()
            t_total += tokens.size(0)
            train_bar.set_postfix(acc=f"{t_corr/t_total:.3f}")

        # Validation
        model.eval()
        v_loss, v_corr, v_total = 0, 0, 0
        with torch.no_grad():
            for tokens, labels in val_loader:
                tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
                logits = model(tokens)
                loss = criterion(logits, labels)
                v_loss += loss.item() * tokens.size(0)
                v_corr += (logits.argmax(1) == labels).sum().item()
                v_total += tokens.size(0)

        # Metrics
        results = {
            "train_loss": t_loss/t_total, "train_acc": t_corr/t_total,
            "val_loss": v_loss/v_total, "val_acc": v_corr/v_total
        }
        
        # Log to DB
        db.log_metrics(exp_id, epoch, results["train_loss"], results["val_loss"], 
                       results["train_acc"], results["val_acc"])
        
        # Scheduler Step
        scheduler.step()

        print(f"Epoch {epoch:02d}: Val Acc={results['val_acc']:.4f} | LR={optimizer.param_groups[0]['lr']:.6f}")

        if results["val_acc"] > best_val_acc:
            best_val_acc = results["val_acc"]
            torch.save(model.state_dict(), f"{model_name}_best.pt")
            # Update best accuracy in DB
            with db._get_connection() as conn:
                conn.execute("UPDATE experiments SET best_val_acc = ? WHERE id = ?", (best_val_acc, exp_id))

    total_time = time.time() - start_time
    with db._get_connection() as conn:
        conn.execute("UPDATE experiments SET total_time = ? WHERE id = ?", (total_time, exp_id))
    
    return best_val_acc

# ─────────────────────────────────────────────────────────────────────────────
# 3. Execution
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("-" * 50)
    print("QUANTUM NLP: Advanced Training Pipeline")
    print("-" * 50)

    # Load from DB
    df_train = db.load_dataset(split='train')
    df_val   = db.load_dataset(split='validation')

    if df_train.empty:
        print("[Error] DB is empty! Run ingest_data.py first.")
        exit(1)

    train_sentences = df_train['sentence'].tolist()
    train_labels    = df_train['label'].tolist()
    val_sentences   = df_val['sentence'].tolist()
    val_labels      = df_val['label'].tolist()

    # 1. BUILD/SAVE VOCAB
    print("[Vocab] Building...")
    counter = Counter()
    for s in train_sentences:
        counter.update(str(s).lower().split())
    most_common = counter.most_common(VOCAB_SIZE - 2)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for idx, (word, _) in enumerate(most_common, start=2):
        vocab[word] = idx
    db.save_vocab(vocab)

    # 2. CREATE LOADERS
    train_ds = SSTDataset(train_sentences, train_labels, vocab)
    val_ds   = SSTDataset(val_sentences, val_labels, vocab)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_VAL)

    # Subsampled Loader for Quantum
    q_train_ds = SSTDataset(train_sentences[:QUANTUM_TRAIN_SUBSET], train_labels[:QUANTUM_TRAIN_SUBSET], vocab)
    q_train_loader = DataLoader(q_train_ds, batch_size=BATCH_TRAIN, shuffle=True)

    # 3. TRAIN
    print("\n--- Phase 1: Quantum Transformer ---")
    q_model = QuantumTransformer(quantum_stride=QUANTUM_STRIDE)
    train_model(q_model, q_train_loader, val_loader, QUANTUM_EPOCHS, "quantum")

    print("\n--- Phase 2: Classical Transformer ---")
    c_model = ClassicalTransformer()
    train_model(c_model, train_loader, val_loader, EPOCHS, "classical")

    print("\n[OK] Training complete! Models and metrics are in the database.")

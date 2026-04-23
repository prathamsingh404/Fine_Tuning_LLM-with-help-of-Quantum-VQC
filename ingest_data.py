"""
ingest_data.py
==============
One-time script to migrate data from JSON files and Hugging Face 
into the new SQLite database.
"""

import json
import os
from datasets import load_dataset
from database import DatabaseManager

def main():
    db = DatabaseManager()
    print("--- Starting Data Ingestion ---")

    # 1. Ingest Vocabulary
    if os.path.exists("vocab.json"):
        print("[Vocab] Migrating vocab.json -> DB...")
        with open("vocab.json", "r", encoding="utf-8") as f:
            vocab = json.load(f)
        db.save_vocab(vocab)
        print(f"[OK] Migrated {len(vocab):,} tokens")
    else:
        print("[Vocab] No vocab.json found. Skipping.")

    # 2. Ingest Training Histories
    for model_name in ["quantum", "classical"]:
        hist_path = f"{model_name}_history.json"
        if os.path.exists(hist_path):
            print(f"[History] Migrating {hist_path} -> DB...")
            with open(hist_path, "r") as f:
                hist = json.load(f)
            
            # Log experiment
            exp_id = db.log_experiment(
                model_name=model_name,
                params=hist.get("params", 0),
                best_acc=hist.get("best_val_acc", 0.0),
                total_time=hist.get("total_time", 0.0)
            )

            # Log epoch metrics
            for i in range(len(hist["train_loss"])):
                db.log_metrics(
                    experiment_id=exp_id,
                    epoch=i + 1,
                    train_loss=hist["train_loss"][i],
                    val_loss=hist["val_loss"][i],
                    train_acc=hist["train_acc"][i],
                    val_acc=hist["val_acc"][i]
                )
            print(f"[OK] Migrated {model_name} history")
        else:
            print(f"[History] {hist_path} not found. Skipping.")

    # 3. Ingest SST-2 Dataset (from HF)
    print("[Dataset] Loading SST-2 from Hugging Face...")
    try:
        dataset = load_dataset("glue", "sst2")
        
        # Check if already ingested to avoid duplicates
        with db._get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM dataset").fetchone()[0]
        
        if count == 0:
            for split in ["train", "validation"]:
                sentences = dataset[split]["sentence"]
                labels = dataset[split]["label"]
                print(f"[Dataset] Saving {split} split ({len(sentences):,} rows)...")
                db.save_dataset(sentences, labels, split=split)
            print("[OK] Dataset ingestion complete")
        else:
            print("[Dataset] Already ingested. Skipping.")
            
    except Exception as e:
        print(f"[Error] Failed to ingest dataset: {e}")

    print("\n--- Ingestion Finished ---")

if __name__ == "__main__":
    main()

import time
import os
from database import DatabaseManager

def log_metrics_many(self, experiment_id, metrics_list):
    """Log metrics for multiple epochs efficiently."""
    with self._get_connection() as conn:
        cursor = conn.cursor()
        data = [(experiment_id, m["epoch"], m["train_loss"], m["val_loss"], m["train_acc"], m["val_acc"]) for m in metrics_list]
        cursor.executemany("""
            INSERT INTO metrics (experiment_id, epoch, train_loss, val_loss, train_acc, val_acc)
            VALUES (?, ?, ?, ?, ?, ?)
        """, data)
        conn.commit()

DatabaseManager.log_metrics_many = log_metrics_many

def main():
    db_path = "benchmark_test.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    db = DatabaseManager(db_path)

    # Prepare dummy data
    N = 1000
    hist = {
        "train_loss": [0.5] * N,
        "val_loss": [0.6] * N,
        "train_acc": [0.8] * N,
        "val_acc": [0.7] * N,
    }
    exp_id = 1

    # 2. Optimized: Batch Inserts
    start_time = time.time()
    metrics_list = []
    for i in range(len(hist["train_loss"])):
        metrics_list.append({
            "epoch": i + 1,
            "train_loss": hist["train_loss"][i],
            "val_loss": hist["val_loss"][i],
            "train_acc": hist["train_acc"][i],
            "val_acc": hist["val_acc"][i]
        })
    db.log_metrics_many(exp_id, metrics_list)
    optimized_time = time.time() - start_time
    print(f"Optimized (Batch inserts): {optimized_time:.4f} seconds")

    if os.path.exists(db_path):
        os.remove(db_path)

if __name__ == "__main__":
    main()

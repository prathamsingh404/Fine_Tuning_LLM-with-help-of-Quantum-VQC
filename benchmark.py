import time
import os
from database import DatabaseManager

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

    # 1. Baseline: N+1 Inserts
    start_time = time.time()
    for i in range(len(hist["train_loss"])):
        db.log_metrics(
            experiment_id=exp_id,
            epoch=i + 1,
            train_loss=hist["train_loss"][i],
            val_loss=hist["val_loss"][i],
            train_acc=hist["train_acc"][i],
            val_acc=hist["val_acc"][i]
        )
    baseline_time = time.time() - start_time
    print(f"Baseline (N+1 inserts): {baseline_time:.4f} seconds")

    if os.path.exists(db_path):
        os.remove(db_path)

if __name__ == "__main__":
    main()

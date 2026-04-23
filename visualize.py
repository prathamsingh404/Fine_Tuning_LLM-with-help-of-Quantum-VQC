"""
visualize.py
============
Generates plots by pulling recent experiment data from the SQLite database.
"""

import os
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from model import QuantumTransformer, ClassicalTransformer
from database import DatabaseManager

# Global Style
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

CLR_QUANTUM   = "#7C3AED"
CLR_CLASSICAL = "#0D9488"

db = DatabaseManager()

def get_latest_histories():
    """Fetch the most recent Quantum and Classical experiments."""
    with db._get_connection() as conn:
        q_exp = pd.read_sql_query("SELECT id, best_val_acc, params FROM experiments WHERE model_name='quantum' ORDER BY id DESC LIMIT 1", conn)
        c_exp = pd.read_sql_query("SELECT id, best_val_acc, params FROM experiments WHERE model_name='classical' ORDER BY id DESC LIMIT 1", conn)
        
        if q_exp.empty or c_exp.empty:
            return None, None
            
        q_id = q_exp.iloc[0]['id']
        c_id = c_exp.iloc[0]['id']
        
        q_metrics = pd.read_sql_query(f"SELECT * FROM metrics WHERE experiment_id={q_id} ORDER BY epoch ASC", conn)
        c_metrics = pd.read_sql_query(f"SELECT * FROM metrics WHERE experiment_id={c_id} ORDER BY epoch ASC", conn)
        
        return {
            "metrics": q_metrics, "best_val_acc": q_exp.iloc[0]['best_val_acc'], "params": q_exp.iloc[0]['params']
        }, {
            "metrics": c_metrics, "best_val_acc": c_exp.iloc[0]['best_val_acc'], "params": c_exp.iloc[0]['params']
        }

def plot_curves(q_data, c_data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss
    ax1.plot(q_data["metrics"]["epoch"], q_data["metrics"]["train_loss"], color=CLR_QUANTUM, label="Q-Train")
    ax1.plot(q_data["metrics"]["epoch"], q_data["metrics"]["val_loss"], color=CLR_QUANTUM, linestyle="--", label="Q-Val")
    ax1.plot(c_data["metrics"]["epoch"], c_data["metrics"]["train_loss"], color=CLR_CLASSICAL, label="C-Train")
    ax1.plot(c_data["metrics"]["epoch"], c_data["metrics"]["val_loss"], color=CLR_CLASSICAL, linestyle="--", label="C-Val")
    ax1.set_title("Loss Curves", fontweight="bold")
    ax1.legend()
    
    # Accuracy
    ax2.plot(q_data["metrics"]["epoch"], q_data["metrics"]["val_acc"], color=CLR_QUANTUM, marker="o", label="Quantum")
    ax2.plot(c_data["metrics"]["epoch"], c_data["metrics"]["val_acc"], color=CLR_CLASSICAL, marker="s", label="Classical")
    ax2.set_title("Validation Accuracy", fontweight="bold")
    ax2.set_ylim(0.4, 1.0)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("accuracy_curves.png")
    plt.close()

def plot_params(q_data, c_data):
    categories = ["Total Parameters"]
    q_vals = [q_data["params"]]
    c_vals = [c_data["params"]]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, q_vals, width, label="Quantum", color=CLR_QUANTUM)
    ax.bar(x + width/2, c_vals, width, label="Classical", color=CLR_CLASSICAL)
    ax.set_title("Parameter Comparison", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("parameters.png")
    plt.close()

if __name__ == "__main__":
    q_data, c_data = get_latest_histories()
    if q_data:
        plot_curves(q_data, c_data)
        plot_params(q_data, c_data)
        print("[OK] Visualization files updated from DB.")
    else:
        print("[Error] No experiments found in DB.")

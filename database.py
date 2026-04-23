"""
database.py
===========
SQLite Database Management for Quantum NLP Project.
Handles schema creation, persistent storage of vocabulary, experiments, 
training metrics, inference logs, and the SST-2 dataset.
"""

import sqlite3
import json
import os
from datetime import datetime
import pandas as pd

DB_PATH = "quantum_nlp.db"

class DatabaseManager:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initialize the database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Vocabulary Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vocabulary (
                    word TEXT PRIMARY KEY,
                    idx INTEGER NOT NULL
                )
            """)

            # 2. Experiments Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    params INTEGER,
                    best_val_acc REAL,
                    total_time REAL,
                    config JSON
                )
            """)

            # 3. Training Metrics Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    epoch INTEGER,
                    train_loss REAL,
                    val_loss REAL,
                    train_acc REAL,
                    val_acc REAL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)

            # 4. Predictions (Inference) Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    input_text TEXT,
                    model_name TEXT,
                    prediction TEXT,
                    confidence REAL,
                    latency REAL
                )
            """)

            # 5. Dataset Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dataset (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sentence TEXT,
                    label INTEGER,
                    split TEXT -- 'train', 'validation', 'test'
                )
            """)
            conn.commit()

    # --- Vocabulary Methods ---
    def save_vocab(self, vocab: dict):
        """Save word->idx mapping to DB."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM vocabulary")
            cursor.executemany(
                "INSERT INTO vocabulary (word, idx) VALUES (?, ?)",
                vocab.items()
            )
            conn.commit()

    def load_vocab(self) -> dict:
        """Load vocabulary from DB."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT word, idx FROM vocabulary")
            return {row[0]: row[1] for row in cursor.fetchall()}

    # --- Experiment Methods ---
    def log_experiment(self, model_name, params, best_acc, total_time, config=None):
        """Log a completed experiment."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO experiments (model_name, params, best_val_acc, total_time, config)
                VALUES (?, ?, ?, ?, ?)
            """, (model_name, params, best_acc, total_time, json.dumps(config) if config else None))
            conn.commit()
            return cursor.lastrowid

    def log_metrics(self, experiment_id, epoch, train_loss, val_loss, train_acc, val_acc):
        """Log metrics for a single epoch."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO metrics (experiment_id, epoch, train_loss, val_loss, train_acc, val_acc)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (experiment_id, epoch, train_loss, val_loss, train_acc, val_acc))
            conn.commit()

    # --- Prediction Methods ---
    def log_prediction(self, input_text, model_name, prediction, confidence, latency=0.0):
        """Log an inference request."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (input_text, model_name, prediction, confidence, latency)
                VALUES (?, ?, ?, ?, ?)
            """, (input_text, model_name, prediction, confidence, latency))
            conn.commit()

    def get_prediction_history(self, limit=50):
        """Retrieve recent predictions."""
        with self._get_connection() as conn:
            return pd.read_sql_query(
                "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?", 
                conn, params=(limit,)
            )

    # --- Dataset Methods ---
    def save_dataset(self, sentences, labels, split='train'):
        """Save a dataset split to DB."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            data = [(s, l, split) for s, l in zip(sentences, labels)]
            cursor.executemany(
                "INSERT INTO dataset (sentence, label, split) VALUES (?, ?, ?)",
                data
            )
            conn.commit()

    def load_dataset(self, split='train'):
        """Load a dataset split from DB."""
        with self._get_connection() as conn:
            return pd.read_sql_query(
                "SELECT sentence, label FROM dataset WHERE split = ?",
                conn, params=(split,)
            )

if __name__ == "__main__":
    db = DatabaseManager()
    print(f"Database initialized at {DB_PATH}")

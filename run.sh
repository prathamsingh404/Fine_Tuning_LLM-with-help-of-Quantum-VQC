#!/usr/bin/env bash
# run.sh
# ======
# One-shot script to install dependencies, train both models,
# generate visualizations, and launch the Streamlit demo.
#
# Usage (Linux / macOS / Git Bash on Windows):
#     chmod +x run.sh
#     ./run.sh
#
# On Windows PowerShell, run each command manually (see README.md).

set -e   # exit immediately if any command fails
echo "=================================================="
echo " Quantum-Enhanced LLM Fine-Tuning — Runner Script"
echo "=================================================="

# ── Step 1: Install dependencies ──────────────────────────────────────────────
echo ""
echo "[Step 1/4] Installing Python dependencies ..."
pip install -r requirements.txt

# ── Step 2: Train both models ──────────────────────────────────────────────────
echo ""
echo "[Step 2/4] Training QuantumTransformer and ClassicalTransformer ..."
echo "           This will take ~30-60 minutes depending on your hardware."
python train.py

# ── Step 3: Generate visualizations ───────────────────────────────────────────
echo ""
echo "[Step 3/4] Generating visualisation plots ..."
python visualize.py

# ── Step 4: Launch Streamlit app ───────────────────────────────────────────────
echo ""
echo "[Step 4/4] Launching Streamlit demo at http://localhost:8501 ..."
streamlit run app.py

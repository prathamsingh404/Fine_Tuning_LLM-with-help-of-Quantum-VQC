"""
app.py
======
Enhanced Streamlit UI for Quantum Sentiment Analysis.
Features SQLite history, advanced QML visualizations, and live performance metrics.
"""

import os
import json
import torch
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from model import QuantumTransformer, ClassicalTransformer
from database import DatabaseManager
import time

# --- Setup ---
DEVICE = torch.device("cpu")
db = DatabaseManager()

st.set_page_config(page_title="Quantum NLP Engine", page_icon="⚛️", layout="wide")

# --- Custom Styles ---
st.markdown("""
<style>
    .stApp { background-color: #0b0e14; color: #e0e0e0; }
    .hero {
        background: linear-gradient(135deg, #1e1b4b 0%, #4c1d95 50%, #0d9488 100%);
        padding: 3rem; border-radius: 20px; text-align: center; margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .hero h1 { font-size: 3.5rem; font-weight: 800; color: white; margin: 0; }
    .card {
        background: #161b22; border: 1px solid #30363d; border-radius: 12px;
        padding: 1.5rem; margin-bottom: 1rem;
    }
    .stat-label { color: #8b949e; font-size: 0.9rem; }
    .stat-value { font-size: 1.5rem; font-weight: 700; color: #58a6ff; }
</style>
""", unsafe_allow_html=True)

# --- Resource Loading ---
@st.cache_resource
def get_models():
    q_model = QuantumTransformer()
    c_model = ClassicalTransformer()
    if os.path.exists("quantum_best.pt"):
        q_model.load_state_dict(torch.load("quantum_best.pt", map_location=DEVICE))
    if os.path.exists("classical_best.pt"):
        c_model.load_state_dict(torch.load("classical_best.pt", map_location=DEVICE))
    q_model.eval(); c_model.eval()
    return q_model, c_model

@st.cache_resource
def get_vocab():
    v = db.load_vocab()
    if not v:
        # Fallback to file if DB is empty but file exists
        if os.path.exists("vocab.json"):
            with open("vocab.json", "r") as f:
                v = json.load(f)
                db.save_vocab(v)
    return v

# --- Hero ---
st.markdown('<div class="hero"><h1>Quantum NLP Engine v2.0</h1><p>Active Learning with Data Re-uploading & Residual VQC</p></div>', unsafe_allow_html=True)

vocab = get_vocab()
q_model, c_model = get_models()

tab_demo, tab_history, tab_theory = st.tabs(["🚀 Live Inference", "📜 History", "🧬 Architecture"])

# --- Tab 1: Live Demo ---
with tab_demo:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### ✍️ Input Analysis")
        input_text = st.text_area("Review Text", "The experimental results were surprisingly promising and robust.", height=150)
        
        if st.button("Analyze with Quantum & Classical", type="primary"):
            if not vocab:
                st.error("Vocabulary not found. Please train models first.")
            else:
                # Tokenize
                words = input_text.lower().split()
                tokens = [vocab.get(w, 1) for w in words]
                tokens = tokens[:128]
                tokens += [0] * (128 - len(tokens))
                token_tensor = torch.tensor([tokens], dtype=torch.long)
                
                # Predict
                start = time.time()
                with torch.no_grad():
                    q_logits = q_model(token_tensor)
                    q_probs = torch.softmax(q_logits, dim=1)[0]
                q_latency = time.time() - start
                
                start = time.time()
                with torch.no_grad():
                    c_logits = c_model(token_tensor)
                    c_probs = torch.softmax(c_logits, dim=1)[0]
                c_latency = time.time() - start
                
                # Log to DB
                q_label = "Positive" if q_probs[1] > 0.5 else "Negative"
                db.log_prediction(input_text, "quantum", q_label, float(q_probs.max()), q_latency)
                
                # Results UI
                res_q, res_c = st.columns(2)
                with res_q:
                    st.markdown(f"#### ⚛️ Quantum")
                    st.progress(float(q_probs[1]), text=f"Positive: {q_probs[1]*100:.1f}%")
                    st.caption(f"Latency: {q_latency*1000:.1f}ms")
                with res_c:
                    st.markdown(f"#### 🤖 Classical")
                    st.progress(float(c_probs[1]), text=f"Positive: {c_probs[1]*100:.1f}%")
                    st.caption(f"Latency: {c_latency*1000:.1f}ms")

    with col2:
        st.markdown("### 📊 Metrics")
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<span class="stat-label">Dataset</span><br><span class="stat-value">SST-2 (GLUE)</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<span class="stat-label">Quantum Layers</span><br><span class="stat-value">3 (Re-uploading)</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<span class="stat-label">Residual Connections</span><br><span class="stat-value">Enabled</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# --- Tab 2: History ---
with tab_history:
    st.markdown("### 📜 Prediction History")
    history_df = db.get_prediction_history(50)
    if not history_df.empty:
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No predictions logged yet.")
        
    st.markdown("### 📈 Training Experiments")
    with db._get_connection() as conn:
        exp_df = pd.read_sql_query("SELECT * FROM experiments ORDER BY id DESC", conn)
    st.table(exp_df)

# --- Tab 3: Theory ---
with tab_theory:
    st.markdown("### 🧬 Quantum Circuit Evolution")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Old Architecture (v1):**
        - Single Variational Layer
        - Simple CNOT Ring
        - Linear Baseline
        
        **New Architecture (v2):**
        - **3-5 Variational Layers** (Deep VQC)
        - **Data Re-uploading**: The classical features are injected multiple times between layers to increase expressivity.
        - **Strong Entanglement**: All-to-all connectivity for better feature interaction.
        - **Residual Connection**: $y = x + VQC(x)$ avoids gradient vanishing ("Barren Plateaus").
        """)
    with col_b:
        st.code("""
# Data Re-uploading Loop
for layer in range(N_LAYERS):
    encoding_layer(inputs)    # Injected EVERY layer
    variational_layer(weights) # Trainable
    entanglement_layer()       # CNOTs
        """, language="python")

    st.markdown("### 📉 Latest Training Trace")
    if os.path.exists("accuracy_curves.png"):
        st.image("accuracy_curves.png")
    else:
        st.warning("Run 'python visualize.py' to generate latest plots.")

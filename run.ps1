# run.ps1 — Windows PowerShell script to run the full project pipeline
# Usage: Right-click → "Run with PowerShell"  OR  type: .\run.ps1

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host " Quantum-Enhanced LLM Fine-Tuning — PowerShell Runner" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# Step 1: Install dependencies
Write-Host "`n[Step 1/4] Installing Python dependencies ..." -ForegroundColor Yellow
python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] pip install failed!" -ForegroundColor Red; exit 1 }

# Step 2: Train both models
Write-Host "`n[Step 2/4] Training models (this takes 30-90 min on CPU) ..." -ForegroundColor Yellow
python train.py
if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] train.py failed!" -ForegroundColor Red; exit 1 }

# Step 3: Generate plots
Write-Host "`n[Step 3/4] Generating visualisation plots ..." -ForegroundColor Yellow
python visualize.py
if ($LASTEXITCODE -ne 0) { Write-Host "[ERROR] visualize.py failed!" -ForegroundColor Red; exit 1 }

# Step 4: Launch Streamlit
Write-Host "`n[Step 4/4] Launching Streamlit at http://localhost:8501 ..." -ForegroundColor Green
python -m streamlit run app.py

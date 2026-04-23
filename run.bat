@echo off
REM run.bat — Windows batch script to run the full project pipeline
REM Usage: Double-click this file OR run it in Command Prompt / PowerShell

echo ==================================================
echo  Quantum-Enhanced LLM Fine-Tuning — Windows Runner
echo ==================================================

echo.
echo [Step 1/4] Installing Python dependencies ...
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] pip install failed. Make sure Python is installed.
    pause
    exit /b 1
)

echo.
echo [Step 2/4] Training QuantumTransformer and ClassicalTransformer ...
echo            This will take 30-90 minutes on CPU. Please be patient.
python train.py
if %errorlevel% neq 0 (
    echo [ERROR] train.py failed. Check the error above.
    pause
    exit /b 1
)

echo.
echo [Step 3/4] Generating visualisation plots ...
python visualize.py
if %errorlevel% neq 0 (
    echo [ERROR] visualize.py failed. Check the error above.
    pause
    exit /b 1
)

echo.
echo [Step 4/4] Launching Streamlit demo at http://localhost:8501 ...
python -m streamlit run app.py

pause

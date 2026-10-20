@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   KD-LowData: Auto-Runner for NVIDIA RTX 4060 Laptop
echo ============================================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to PATH.
    echo Please install Python 3.10, 3.11, or 3.12 from python.org
    echo and make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [1/5] Setting up virtual environment (.venv)...
if not exist .venv (
    python -m venv .venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

echo [2/5] Activating virtual environment...
call .venv\Scripts\activate

echo [3/5] Installing PyTorch with CUDA 12.1 acceleration...
python -m pip install -q --upgrade pip
python -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -q -r requirements.txt

echo [4/5] Checking GPU detection...
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Active GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (WARNING)')"
echo.

echo ============================================================
echo   Starting Training Grid (28 runs, ~1.5 to 2 hours total)...
echo   CIFAR-100 will auto-download on the first run.
echo ============================================================
python src/run_grid.py --config configs/kaggle_fast.yaml --num_workers 0

echo.
echo ============================================================
echo   Generating Final Graphs and Summary Tables...
echo ============================================================
python src/plot_results.py

echo.
echo ============================================================
echo   SUCCESS! Training complete!
echo   All plots, tables, and metrics are saved in:
echo   - results/ (folder)
echo   - kd_results.zip (zipped archive)
echo.
echo   Send 'kd_results.zip' back to Yusuf!
echo ============================================================
pause

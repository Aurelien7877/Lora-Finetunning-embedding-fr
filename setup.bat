@echo off
echo ==========================================
echo  SETUP CLEAN MTEB EMBEDDING (RTX 3070)
echo ==========================================
echo.

:: 1. Clean env
echo [1/8] Nettoyage ancien venv...
call deactivate 2>nul
rmdir /s /q .venv 2>nul

:: 2. Create venv
echo [2/8] Creation venv Python 3.10...
py -3.10 -m venv .venv

:: 3. Activate
echo [3/8] Activation venv...
call .venv\Scripts\activate.bat

:: 4. Pip upgrade
echo [4/8] Upgrade pip...
python -m pip install --upgrade pip setuptools wheel

:: 5. PyTorch CUDA 12.1
echo [5/8] Installation PyTorch CUDA 12.1...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

:: 6. Core ML stack (LOCKED VERSIONS)
echo [6/8] Installation transformers + sentence-transformers...
pip install -U transformers>=4.48.0
pip install -U sentence-transformers>=3.0.0
pip install -U accelerate datasets scikit-learn tqdm
pip install -U huggingface_hub safetensors

:: 7. Training + eval stack
echo [7/8] Installation training stack...
pip install datasets accelerate
pip install scikit-learn numpy scipy tqdm
pip install mteb==1.12.0
pip install faiss-cpu

:: 8. Test import
echo [8/8] Verification...
python -c "from sentence_transformers import SentenceTransformer; print('OK')"

echo.
echo ==========================================
echo  SETUP TERMINÉ
echo ==========================================
pause
@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================================
echo   P-H3-Image WebUI Launcher (Windows)
echo   A1111-Style Easy Interface for PrunaAI / MiniMax-H3
echo ============================================================
echo.

REM 1. Check Git
where git >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Git がインストールされていません。https://git-scm.com/ からインストールしてください。
    pause
    exit /b 1
)

REM 2. Clone ComfyUI engine if missing
if not exist "ComfyUI\main.py" (
    echo [1/4] 推論エンジン ^(ComfyUI^) を取得中...
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git ComfyUI
    if errorlevel 1 (
        echo [ERROR] ComfyUI のクローンに失敗しました。
        pause
        exit /b 1
    )
) else (
    echo [1/4] 推論エンジン ^(ComfyUI^) 確認OK
)

REM 3. Create Python virtual environment if missing
if not exist ".venv\Scripts\python.exe" (
    echo [2/4] Python 仮想環境 ^(.venv^) を構築中...
    where uv >nul 2>nul
    if not errorlevel 1 (
        uv venv .venv --python 3.12
        uv pip install --python .venv\Scripts\python.exe torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
        uv pip install --python .venv\Scripts\python.exe -r ComfyUI\requirements.txt gradio huggingface_hub hf_transfer websocket-client requests
    ) else (
        python -m venv .venv
        .venv\Scripts\python.exe -m pip install --upgrade pip
        .venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
        .venv\Scripts\python.exe -m pip install -r ComfyUI\requirements.txt gradio huggingface_hub hf_transfer websocket-client requests
    )
    if errorlevel 1 (
        echo [ERROR] パッケージのインストールに失敗しました。
        pause
        exit /b 1
    )
) else (
    echo [2/4] Python 仮想環境 ^(.venv^) 確認OK
)

REM 4. Check & Download P-H3-Image quantized weights (24GB VRAM optimal profile)
set "PH3_PROFILE=24gb"
if not "%~1"=="" set "PH3_PROFILE=%~1"

echo [3/4] P-H3-Image モデルファイルをチェック中 ^(Profile: %PH3_PROFILE%^)...
.venv\Scripts\python.exe download_models.py --profile %PH3_PROFILE%
if errorlevel 1 (
    echo [ERROR] モデルのダウンロード中にエラーが発生しました。再実行すると途中から再開します。
    pause
    exit /b 1
)

REM 5. Launch A1111-style WebUI
echo [4/4] P-H3-Image WebUI を起動しています... ^(終了するにはこのウィンドウで Ctrl+C を押してください^)
echo.
.venv\Scripts\python.exe webui.py

pause

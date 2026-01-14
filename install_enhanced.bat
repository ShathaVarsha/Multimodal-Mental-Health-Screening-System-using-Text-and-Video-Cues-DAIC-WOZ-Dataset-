@echo off
REM Quick installation script for Windows Command Prompt users

echo =========================================
echo   ENHANCED FEATURES INSTALLATION
echo =========================================
echo.

echo [1/3] Installing dependencies...
pip install reportlab matplotlib
if %errorlevel% neq 0 (
    echo ERROR: Installation failed
    pause
    exit /b 1
)
echo Done!

echo.
echo [2/3] Creating directories...
if not exist "outputs\reports" mkdir "outputs\reports"
echo Done!

echo.
echo [3/3] Verifying installation...
python -c "import reportlab; import matplotlib; print('All packages OK')"

echo.
echo =========================================
echo   INSTALLATION COMPLETE
echo =========================================
echo.
echo To use enhanced features:
echo   1. Run setup to activate files:
echo      powershell -ExecutionPolicy Bypass -File setup_enhanced.ps1
echo.
echo   2. OR manually copy files:
echo      copy templates\index_new.html templates\index.html
echo      copy step8_web_enhanced.py step8_web_interface.py
echo.
echo   3. Launch server:
echo      python step8_web_interface.py
echo.
pause

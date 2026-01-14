# ==============================================================================
# QUICK START GUIDE
# ==============================================================================
# Follow these steps to set up and run the Depression Screening System
# ==============================================================================

## STEP 1: INSTALL PYTHON PACKAGES
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "STEP 1: Installing Python Packages" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

pip install --upgrade pip
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn flask nltk textblob

## STEP 2: DOWNLOAD NLTK DATA
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "STEP 2: Downloading NLTK Data" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"

## STEP 3: CREATE OUTPUT DIRECTORIES
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "STEP 3: Creating Output Directories" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

New-Item -ItemType Directory -Force -Path "models"
New-Item -ItemType Directory -Force -Path "outputs"
New-Item -ItemType Directory -Force -Path "checkpoints"
New-Item -ItemType Directory -Force -Path "templates"
New-Item -ItemType Directory -Force -Path "static"

Write-Host "`n✓ Directories created" -ForegroundColor Green

## STEP 4: VERIFY DATA FILES
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "STEP 4: Verifying Data Files" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$dataFiles = @(
    "dev_split_Depression_AVEC2017.csv",
    "train_split_Depression_AVEC2017.csv",
    "test_split_Depression_AVEC2017.csv"
)

foreach ($file in $dataFiles) {
    if (Test-Path $file) {
        Write-Host "✓ Found: $file" -ForegroundColor Green
    } else {
        Write-Host "✗ Missing: $file" -ForegroundColor Red
    }
}

# Check session data folders
$sessions = @("300", "301", "302", "303", "304", "305")

foreach ($session in $sessions) {
    $path = "data\$session"
    if (Test-Path $path) {
        Write-Host "✓ Found session: $session" -ForegroundColor Green
    } else {
        Write-Host "✗ Missing session: $session" -ForegroundColor Red
    }
}

## STEP 5: RUN DATA PREPARATION (Optional - Check First)
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "STEP 5: Ready to Train Models" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Setup complete! Next steps:" -ForegroundColor Yellow
Write-Host "`n1. Run full training pipeline:" -ForegroundColor White
Write-Host "   python run_full_pipeline.py" -ForegroundColor Green

Write-Host "`n2. Or run step-by-step:" -ForegroundColor White
Write-Host "   python step1_data_preparation.py" -ForegroundColor Green
Write-Host "   python run_full_pipeline.py  (for remaining steps)" -ForegroundColor Green

Write-Host "`n3. After training, start web interface:" -ForegroundColor White
Write-Host "   python step8_web_interface.py" -ForegroundColor Green

Write-Host "`n4. Open browser:" -ForegroundColor White
Write-Host "   http://127.0.0.1:5000" -ForegroundColor Green

Write-Host "`n========================================`n" -ForegroundColor Cyan

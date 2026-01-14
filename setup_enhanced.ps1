# Quick Setup Script for Enhanced Features
# Run this to install dependencies and set up files

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  ENHANCED FEATURES SETUP" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Install dependencies
Write-Host "[1/4] Installing new dependencies..." -ForegroundColor Yellow
pip install reportlab matplotlib --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "✗ Installation failed" -ForegroundColor Red
    exit 1
}

# Step 2: Backup original files
Write-Host ""
Write-Host "[2/4] Backing up original files..." -ForegroundColor Yellow
if (Test-Path "templates\index.html") {
    Copy-Item "templates\index.html" "templates\index_backup.html" -Force
    Write-Host "✓ Backed up index.html" -ForegroundColor Green
}
if (Test-Path "step8_web_interface.py") {
    Copy-Item "step8_web_interface.py" "step8_web_interface_backup.py" -Force
    Write-Host "✓ Backed up step8_web_interface.py" -ForegroundColor Green
}

# Step 3: Activate new files
Write-Host ""
Write-Host "[3/4] Activating enhanced features..." -ForegroundColor Yellow
if (Test-Path "templates\index_new.html") {
    Copy-Item "templates\index_new.html" "templates\index.html" -Force
    Write-Host "✓ Updated index.html" -ForegroundColor Green
}
if (Test-Path "step8_web_enhanced.py") {
    Copy-Item "step8_web_enhanced.py" "step8_web_interface.py" -Force
    Write-Host "✓ Updated step8_web_interface.py" -ForegroundColor Green
}

# Step 4: Create reports directory
Write-Host ""
Write-Host "[4/4] Creating reports directory..." -ForegroundColor Yellow
if (!(Test-Path "outputs\reports")) {
    New-Item -ItemType Directory -Path "outputs\reports" -Force | Out-Null
    Write-Host "✓ Created outputs/reports/" -ForegroundColor Green
} else {
    Write-Host "✓ Directory already exists" -ForegroundColor Green
}

# Verification
Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  VERIFICATION" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
python -c "import reportlab; import matplotlib; print('✓ All packages installed successfully')"

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "  SETUP COMPLETE!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Run: python step8_web_interface.py" -ForegroundColor White
Write-Host "  2. Open: http://127.0.0.1:5000" -ForegroundColor White
Write-Host "  3. Test camera and PDF features" -ForegroundColor White
Write-Host ""
Write-Host "Read ENHANCED_FEATURES_GUIDE.md for details" -ForegroundColor Cyan
Write-Host ""

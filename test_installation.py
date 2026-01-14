"""
=================================================================
QUICK TEST - Verify Installation
=================================================================
Run this to check if all dependencies are installed correctly
"""

import sys
from pathlib import Path

print("=" * 70)
print("DEPRESSION SCREENING SYSTEM - INSTALLATION TEST")
print("=" * 70)

# Test 1: Python Version
print("\n1. Python Version:")
print(f"   {sys.version}")
if sys.version_info >= (3, 8):
    print("   ✓ Python version OK")
else:
    print("   ✗ Python 3.8+ required")

# Test 2: Required Packages
print("\n2. Required Packages:")

packages = {
    "torch": "PyTorch",
    "transformers": "Hugging Face Transformers",
    "sklearn": "Scikit-learn",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "matplotlib": "Matplotlib",
    "seaborn": "Seaborn",
    "flask": "Flask",
    "nltk": "NLTK"
}

all_installed = True

for pkg, name in packages.items():
    try:
        __import__(pkg)
        print(f"   ✓ {name}")
    except ImportError:
        print(f"   ✗ {name} - NOT INSTALLED")
        all_installed = False

if not all_installed:
    print("\n   ⚠ Missing packages detected!")
    print("   Run: pip install -r requirements.txt")

# Test 3: Project Structure
print("\n3. Project Structure:")

paths = {
    "config.py": "Configuration file",
    "utils.py": "Utility functions",
    "data_preparation_1.py": "Data preparation script",
    "run_full_pipeline.py": "Full training pipeline",
    "web_interface_8.py": "Web interface",
    "templates/index.html": "HTML template",
    "data": "Data directory"
}

for path, desc in paths.items():
    if Path(path).exists():
        print(f"   ✓ {desc}")
    else:
        print(f"   ✗ {desc} - MISSING")

# Test 4: Data Files
print("\n4. Data Files:")

data_files = [
    "dev_split_Depression_AVEC2017.csv",
    "train_split_Depression_AVEC2017.csv",
    "test_split_Depression_AVEC2017.csv"
]

for file in data_files:
    if Path(file).exists():
        print(f"   ✓ {file}")
    else:
        print(f"   ⚠ {file} - Not found")

# Test 5: Session Data
print("\n5. Session Data:")

data_dir = Path("data")
sessions = ["300", "301", "302", "303", "304", "305"]

if data_dir.exists():
    for session in sessions:
        session_path = data_dir / session
        if session_path.exists():
            # Count files
            files = list(session_path.glob("*.txt")) + list(session_path.glob("*.csv"))
            print(f"   ✓ Session {session} ({len(files)} files)")
        else:
            print(f"   ✗ Session {session} - MISSING")
else:
    print("   ✗ data/ directory not found")

# Test 6: Output Directories
print("\n6. Output Directories:")

output_dirs = ["models", "outputs", "checkpoints", "templates", "static"]

for dir_name in output_dirs:
    path = Path(dir_name)
    if path.exists():
        print(f"   ✓ {dir_name}/")
    else:
        print(f"   ℹ {dir_name}/ will be created automatically")

# Test 7: DistilBERT Download Test
print("\n7. DistilBERT Model Test:")

try:
    from transformers import DistilBertTokenizer, DistilBertModel
    
    print("   → Testing DistilBERT download...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    print("   ✓ DistilBERT loaded successfully")
    
    # Test tokenization
    text = "This is a test sentence"
    inputs = tokenizer(text, return_tensors="pt")
    print(f"   ✓ Tokenization works (tokens: {len(inputs['input_ids'][0])})")
    
except Exception as e:
    print(f"   ✗ DistilBERT test failed: {e}")
    print("   ℹ First run may take time to download models")

# Test 8: NLTK Data
print("\n8. NLTK Data:")

try:
    import nltk
    
    nltk_packages = ["punkt", "stopwords", "vader_lexicon"]
    
    for pkg in nltk_packages:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
            print(f"   ✓ {pkg}")
        except LookupError:
            print(f"   ✗ {pkg} - Run: python -c \"import nltk; nltk.download('{pkg}')\"")
except Exception as e:
    print(f"   ✗ NLTK error: {e}")

# Summary
print("\n" + "=" * 70)
print("INSTALLATION TEST COMPLETE")
print("=" * 70)

if all_installed:
    print("\n✓ All critical packages installed!")
    print("\nNext steps:")
    print("  1. Run training: python run_full_pipeline.py")
    print("  2. Or step-by-step: python data_preparation_1.py")
    print("  3. Start web UI: python web_interface_8.py")
else:
    print("\n⚠ Some packages missing. Install with:")
    print("  pip install -r requirements.txt")

print("\n" + "=" * 70 + "\n")

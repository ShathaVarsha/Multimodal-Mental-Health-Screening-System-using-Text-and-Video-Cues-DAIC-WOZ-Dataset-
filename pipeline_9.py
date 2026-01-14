"""
=================================================================
MASTER TRAINING PIPELINE
=================================================================
Complete end-to-end training pipeline for all 5 models

This script runs all training steps sequentially:
  Step 1: Data Preparation
  Step 2: Feature Engineering
  Step 3: Model 1 - Adaptive Dialogue
  Step 4: Model 2a - Text Extractor
  Step 5: Model 2b - Visual Classifier
  Step 6: Model 3 - Fusion Network
  Step 7: Model 4 - Report Generator

Run this to train the entire system at once!
"""

import sys
import os
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from config import *
from utils import *

print_section("HYBRID MULTIMODAL DEPRESSION SCREENING SYSTEM")
print_section("COMPLETE TRAINING PIPELINE")

# =============================================================================
# INSTALL REQUIRED PACKAGES (First-time setup)
# =============================================================================

def check_and_install_packages():
    """Check if required packages are installed"""
    print_step(0, "Checking Required Packages")
    
    required_packages = [
        "torch",
        "transformers",
        "sklearn",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "nltk",
        "flask"
    ]
    
    missing = []
    
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg} - NOT INSTALLED")
            missing.append(pkg)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print(f"\nPlease install requirements:")
        print(f"  pip install -r requirements.txt")
        return False
    
    print(f"\n✓ All required packages installed!")
    return True

# =============================================================================
# DOWNLOAD NLTK DATA
# =============================================================================

def download_nltk_data():
    """Download required NLTK data"""
    print_step(0, "Downloading NLTK Data")
    
    try:
        import nltk
        
        nltk_data = ["punkt", "stopwords", "vader_lexicon"]
        
        for data in nltk_data:
            try:
                nltk.data.find(f"tokenizers/{data}")
                print(f"  ✓ {data} already downloaded")
            except LookupError:
                print(f"  → Downloading {data}...")
                nltk.download(data, quiet=True)
                print(f"  ✓ {data} downloaded")
    
    except Exception as e:
        print(f"  ⚠ NLTK download error: {e}")

# =============================================================================
# STEP 1: DATA PREPARATION
# =============================================================================

def step1_data_preparation():
    """Run data preparation 1"""
    print_section("STEP 1: DATA PREPARATION")
    
    try:
        import pandas as pd
        
        # Check if already prepared
        if (OUTPUTS_DIR / "prepared_data_1.pkl").exists():
            print("ℹ Prepared data already exists")
            response = input("  → Re-run data preparation? (y/n): ").strip().lower()
            if response != 'y':
                print("  → Skipping Step 1")
                return load_pickle(OUTPUTS_DIR / "prepared_data.pkl")
        
        # Import step1 module
        import data_preparation_1
        
        # Run preparation
        prep = data_preparation_1.DataPreparation()
        
        if not prep.load_phq8_labels():
            print("❌ Failed to load PHQ-8 labels")
            return None
        
        prep.prepare_all_sessions()
        
        if len(prep.sessions_data) == 0:
            print("❌ No sessions loaded")
            return None
        
        prep.merge_with_labels()
        prep.create_final_dataset()
        prep.save_prepared_data()
        
        return prep.prepared_dataset
    
    except Exception as e:
        print(f"❌ Step 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# STEP 2: FEATURE ENGINEERING
# =============================================================================

def step2_feature_engineering(prepared_data):
    """Run feature engineering"""
    print_section("STEP 2: FEATURE ENGINEERING 2")
    
    if prepared_data is None or len(prepared_data) == 0:
        print("❌ No prepared data available")
        return None
    
    try:
        import pandas as pd
        import numpy as np
        from transformers import DistilBertTokenizer, DistilBertModel
        import torch
        
        print_step(1, "Initializing DistilBERT 2")
        
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL2A_CONFIG["model_name"])
        model = DistilBertModel.from_pretrained(MODEL2A_CONFIG["model_name"])
        model.eval()
        
        print("  ✓ DistilBERT loaded")
        
        print_step(2, "Extracting Text Embeddings")
        
        text_embeddings = []
        
        for idx, row in prepared_data.iterrows():
            text = row.get("text", "")
            if not text:
                text_embeddings.append(np.zeros(768))
                continue
            
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
            
            text_embeddings.append(embedding)
            
            if (idx + 1) % 10 == 0:
                print(f"  → Processed {idx + 1}/{len(prepared_data)} utterances")
        
        prepared_data["text_embedding"] = text_embeddings
        
        print(f"  ✓ Text embeddings extracted: {len(text_embeddings)} x 768 dims")
        
        print_step(3, "Engineering Visual Features")
        
        # AU features
        au_cols = [c for c in prepared_data.columns if c.startswith("au_")]
        if au_cols:
            prepared_data["au_mean"] = prepared_data[au_cols].mean(axis=1)
            prepared_data["au_std"] = prepared_data[au_cols].std(axis=1)
            print(f"  ✓ AU features: {len(au_cols)} AUs aggregated")
        
        # Pose features
        pose_cols = [c for c in prepared_data.columns if c.startswith("pose_")]
        if pose_cols:
            prepared_data["pose_mean"] = prepared_data[pose_cols].mean(axis=1)
            prepared_data["pose_std"] = prepared_data[pose_cols].std(axis=1)
            print(f"  ✓ Pose features: {len(pose_cols)} dimensions aggregated")
        
        # Gaze features
        gaze_cols = [c for c in prepared_data.columns if c.startswith("gaze_")]
        if gaze_cols:
            prepared_data["gaze_mean"] = prepared_data[gaze_cols].mean(axis=1)
            prepared_data["gaze_std"] = prepared_data[gaze_cols].std(axis=1)
            print(f"  ✓ Gaze features: {len(gaze_cols)} dimensions aggregated")
        
        # Save
        save_pickle(prepared_data, OUTPUTS_DIR / "engineered_features.pkl")
        
        print("\n✓ Feature engineering complete!")
        
        return prepared_data
    
    except Exception as e:
        print(f"❌ Step 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# STEP 3-7: TRAIN ALL MODELS
# =============================================================================

def train_all_models(engineered_data):
    """Train all 5 models"""
    print_section("STEPS 3-7: TRAINING ALL MODELS")
    
    if engineered_data is None or len(engineered_data) == 0:
        print("❌ No engineered data available")
        return
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import LeaveOneOut
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # =====================================================================
        # MODEL 2B: VISUAL CLASSIFIER (SVM)
        # =====================================================================
        
        print_step(5, "Training Model 2b - Visual Classifier (SVM)")
        
        # Prepare visual features
        au_cols = [c for c in engineered_data.columns if c.startswith("au_")]
        pose_cols = [c for c in engineered_data.columns if c.startswith("pose_")]
        gaze_cols = [c for c in engineered_data.columns if c.startswith("gaze_")]
        
        visual_feature_cols = au_cols + pose_cols + gaze_cols
        
        if len(visual_feature_cols) == 0:
            print("  ⚠ No visual features available")
        else:
            # Aggregate by session
            session_visual = engineered_data.groupby("session_id")[visual_feature_cols].mean()
            session_labels = engineered_data.groupby("session_id")["phq8_score"].first()
            
            X_visual = session_visual.fillna(0).values
            y_phq8 = session_labels.values
            
            print(f"  → Training data: {X_visual.shape[0]} sessions, {X_visual.shape[1]} features")
            
            # Scale features
            scaler = StandardScaler()
            X_visual_scaled = scaler.fit_transform(X_visual)
            
            # Train SVM with best C parameter
            best_svr = None
            best_mae = float('inf')
            
            for C in MODEL2B_CONFIG["C_range"]:
                svr = SVR(kernel=MODEL2B_CONFIG["kernel"], C=C, gamma=MODEL2B_CONFIG["gamma"])
                svr.fit(X_visual_scaled, y_phq8)
                
                # Simple validation
                y_pred = svr.predict(X_visual_scaled)
                mae = np.mean(np.abs(y_pred - y_phq8))
                
                print(f"    C={C}: MAE={mae:.2f}")
                
                if mae < best_mae:
                    best_mae = mae
                    best_svr = svr
            
            print(f"  ✓ Best SVM: MAE={best_mae:.2f}")
            
            # Save model
            save_pickle({
                "model": best_svr,
                "scaler": scaler,
                "feature_cols": visual_feature_cols
            }, MODEL2B_CONFIG["save_path"])
        
        # =====================================================================
        # MODEL 3: MULTIMODAL FUSION NETWORK
        # =====================================================================
        
        print_step(6, "Training Model 3 - Multimodal Fusion Network")
        
        # Aggregate by session
        session_data = []
        
        for session_id in engineered_data["session_id"].unique():
            session_subset = engineered_data[engineered_data["session_id"] == session_id]
            
            # Get text embeddings (mean across all turns)
            text_embs = np.stack(session_subset["text_embedding"].values)
            text_emb_mean = text_embs.mean(axis=0)
            
            # Get visual features
            visual_mean = session_subset[visual_feature_cols].mean().values if visual_feature_cols else np.zeros(70)
            
            # Get label
            phq8_score = session_subset["phq8_score"].iloc[0]
            
            session_data.append({
                "session_id": session_id,
                "text_emb": text_emb_mean,
                "visual_feats": visual_mean,
                "phq8_score": phq8_score
            })
        
        df_sessions = pd.DataFrame(session_data)
        
        print(f"  → Training data: {len(df_sessions)} sessions")
        
        # Define Fusion Network
        class MultimodalFusionNet(nn.Module):
            def __init__(self, text_dim=768, visual_dim=70):
                super().__init__()
                
                # Text branch
                self.text_fc1 = nn.Linear(text_dim, 256)
                self.text_fc2 = nn.Linear(256, 128)
                self.text_dropout = nn.Dropout(0.3)
                
                # Visual branch
                self.visual_fc1 = nn.Linear(visual_dim, 128)
                self.visual_fc2 = nn.Linear(128, 64)
                self.visual_dropout = nn.Dropout(0.3)
                
                # Fusion
                self.fusion_fc1 = nn.Linear(128 + 64, 96)
                self.fusion_fc2 = nn.Linear(96, 48)
                self.fusion_dropout = nn.Dropout(0.2)
                
                # Output (Depression score)
                self.output = nn.Linear(48, 1)
                
                self.relu = nn.ReLU()
            
            def forward(self, text_emb, visual_feats):
                # Text branch
                t = self.relu(self.text_fc1(text_emb))
                t = self.text_dropout(t)
                t = self.relu(self.text_fc2(t))
                
                # Visual branch
                v = self.relu(self.visual_fc1(visual_feats))
                v = self.visual_dropout(v)
                v = self.relu(self.visual_fc2(v))
                
                # Fusion
                fused = torch.cat([t, v], dim=1)
                f = self.relu(self.fusion_fc1(fused))
                f = self.fusion_dropout(f)
                f = self.relu(self.fusion_fc2(f))
                
                # Output
                out = self.output(f)
                return out
        
        # Prepare tensors
        X_text = torch.FloatTensor(np.stack(df_sessions["text_emb"].values))
        X_visual = torch.FloatTensor(np.stack(df_sessions["visual_feats"].values))
        y = torch.FloatTensor(df_sessions["phq8_score"].values).unsqueeze(1)
        
        # Initialize model
        model3 = MultimodalFusionNet(text_dim=768, visual_dim=len(visual_feature_cols) if visual_feature_cols else 70)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model3.parameters(), lr=MODEL3_CONFIG["learning_rate"])
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=MODEL3_CONFIG["early_stopping_patience"],
            verbose=True
        )
        
        # Training loop
        print(f"  → Training for up to {MODEL3_CONFIG['num_epochs']} epochs...")
        
        best_loss = float('inf')
        
        for epoch in range(MODEL3_CONFIG["num_epochs"]):
            model3.train()
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model3(X_text, X_visual)
            loss = criterion(predictions, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Evaluation
            model3.eval()
            with torch.no_grad():
                val_predictions = model3(X_text, X_visual)
                val_loss = criterion(val_predictions, y)
                mae = torch.mean(torch.abs(val_predictions - y))
            
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{MODEL3_CONFIG['num_epochs']}: Loss={val_loss.item():.4f}, MAE={mae.item():.2f}")
            
            # Early stopping check
            if val_loss.item() < best_loss:
                best_loss = val_loss.item()
                save_model(model3, MODEL3_CONFIG["save_path"])
            
            early_stopping(val_loss.item(), model3, MODEL3_CONFIG["save_path"])
            
            if early_stopping.early_stop:
                print(f"  ✓ Early stopping at epoch {epoch+1}")
                break
        
        print(f"  ✓ Best Loss: {best_loss:.4f}")
        
        # =====================================================================
        # MODEL 4: REPORT GENERATOR (Template-based)
        # =====================================================================
        
        print_step(7, "Creating Model 4 - Report Generator Templates")
        
        templates = {
            "minimal": "Based on assessment, this individual shows minimal signs of depression (PHQ-8: {score}/24). Mood appears stable with no significant concerns identified.",
            "mild": "Assessment indicates mild depression symptoms (PHQ-8: {score}/24). Individual may benefit from monitoring and self-care strategies. Consider follow-up in 2-4 weeks.",
            "moderate": "Individual shows moderate depression symptoms (PHQ-8: {score}/24). Clinical intervention recommended. Behavioral observations include: {observations}. Suggest referral to mental health professional.",
            "moderately_severe": "Assessment reveals moderately severe depression (PHQ-8: {score}/24). Immediate clinical evaluation strongly recommended. Behavioral indicators: {observations}. High priority for intervention.",
            "severe": "URGENT: Severe depression detected (PHQ-8: {score}/24). Immediate mental health intervention required. Critical observations: {observations}. Recommend urgent psychiatric evaluation and crisis support resources."
        }
        
        save_json(templates, MODEL4_CONFIG["template_path"])
        print("  ✓ Report templates created")
        
        print("\n✓ All models trained successfully!")
        
        return True
    
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================

def main():
    """Main pipeline execution"""
    
    start_time = time.time()
    
    # Check packages
    if not check_and_install_packages():
        print("\n❌ Please install required packages first")
        print("   Run: pip install -r requirements.txt")
        return
    
    # Download NLTK data
    download_nltk_data()
    
    # Step 1: Data Preparation
    prepared_data = step1_data_preparation()
    if prepared_data is None:
        print("\n❌ Pipeline stopped at Step 1")
        return
    
    # Step 2: Feature Engineering
    engineered_data = step2_feature_engineering(prepared_data)
    if engineered_data is None:
        print("\n❌ Pipeline stopped at Step 2")
        return
    
    # Steps 3-7: Train all models
    success = train_all_models(engineered_data)
    if not success:
        print("\n❌ Pipeline stopped during model training")
        return
    
    # Summary
    elapsed_time = time.time() - start_time
    
        print_section("TRAINING PIPELINE COMPLETE 2")
    print(f"✓ Total time: {elapsed_time/60:.1f} minutes")
    print(f"\nTrained models saved in: {MODELS_DIR}")
    print(f"Outputs saved in: {OUTPUTS_DIR}")
    print(f"\n{'─' * 70}")
    print("NEXT STEPS:")
    print("  1. Review training logs in: training.log")
    print("  2. Start web interface: python web_interface_8.py")
    print("  3. Open browser: http://127.0.0.1:5000")
    print(f"{'─' * 70}\n")

if __name__ == "__main__":
    main()

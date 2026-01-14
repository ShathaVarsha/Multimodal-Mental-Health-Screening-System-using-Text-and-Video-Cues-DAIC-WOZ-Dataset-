"""
=================================================================
STEP 2: FEATURE ENGINEERING
=================================================================
Extract and engineer features from prepared data

This script:
1. Loads prepared data from Step 1
2. Extracts text embeddings using DistilBERT (768-dim)
3. Engineers visual features (AUs, pose, gaze)
4. Computes sentiment features
5. Creates aggregated session-level features
6. Saves engineered features for model training

Inputs:
  - outputs/prepared_data.pkl (from Step 1)

Outputs:
  - outputs/engineered_features.pkl (complete feature set)
  - outputs/text_embeddings.pkl (DistilBERT embeddings)
  - outputs/visual_features.pkl (engineered visual features)
  - outputs/sentiment_features.pkl (sentiment scores)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from config import *
from utils import *

# Check if transformers is installed
try:
    from transformers import DistilBertTokenizer, DistilBertModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠ transformers not installed. Install with: pip install transformers")
    TRANSFORMERS_AVAILABLE = False

# Check if nltk is installed
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    print("⚠ nltk not installed. Install with: pip install nltk")
    NLTK_AVAILABLE = False

# =============================================================================
# FEATURE ENGINEERING CLASS
# =============================================================================

class FeatureEngineering:
    """Handles all feature extraction and engineering"""
    
    def __init__(self):
        """Initialize feature engineering"""
        self.logger = setup_logging(LOG_FILE, LOG_LEVEL)
        print_section("STEP 2: FEATURE ENGINEERING")
        
        self.prepared_data = None
        self.engineered_data = None
        
        # Initialize DistilBERT
        self.tokenizer = None
        self.text_model = None
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = None
        
    def load_prepared_data(self):
        """Load prepared data from Step 1"""
        print_step(1, "Loading Prepared Data")
        
        prepared_file = OUTPUTS_DIR / "prepared_data.pkl"
        
        if not prepared_file.exists():
            print(f"❌ Prepared data not found: {prepared_file}")
            print(f"   Please run: python data_preparation_1.py")
            return False
        
        self.prepared_data = load_pickle(prepared_file)
        
        print(f"✓ Loaded prepared data")
        print(f"  Shape: {self.prepared_data.shape}")
        print(f"  Sessions: {self.prepared_data['session_id'].nunique()}")
        print(f"  Total turns: {len(self.prepared_data)}")
        
        return True
    
    def initialize_text_models(self):
        """Initialize DistilBERT and sentiment analyzer"""
        print_step(2, "Initializing Text Processing Models")
        
        if not TRANSFORMERS_AVAILABLE:
            print("❌ transformers not available")
            return False
        
        try:
            # Load DistilBERT
            print("  → Loading DistilBERT (may take time on first run)...")
            self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL2A_CONFIG["model_name"])
            self.text_model = DistilBertModel.from_pretrained(MODEL2A_CONFIG["model_name"])
            self.text_model.eval()
            print("  ✓ DistilBERT loaded")
            
            # Load sentiment analyzer
            if NLTK_AVAILABLE:
                try:
                    self.sentiment_analyzer = SentimentIntensityAnalyzer()
                    print("  ✓ VADER sentiment analyzer loaded")
                except:
                    print("  ⚠ VADER not available. Download with:")
                    print("     python -c \"import nltk; nltk.download('vader_lexicon')\"")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            return False
    
    def extract_text_embeddings(self):
        """Extract DistilBERT embeddings for all utterances"""
        print_step(3, "Extracting Text Embeddings (DistilBERT)")
        
        if self.tokenizer is None or self.text_model is None:
            print("❌ DistilBERT not initialized")
            return
        
        text_embeddings = []
        
        total = len(self.prepared_data)
        
        for idx, row in self.prepared_data.iterrows():
            text = row.get("text", "")
            
            # Handle empty text
            if not text or pd.isna(text):
                text_embeddings.append(np.zeros(MODEL2A_CONFIG["embedding_dim"]))
                continue
            
            # Clean text
            text = clean_text(str(text))
            
            try:
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MODEL2A_CONFIG["max_length"],
                    padding=True
                )
                
                # Get embeddings (CLS token)
                with torch.no_grad():
                    outputs = self.text_model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
                
                text_embeddings.append(embedding)
                
            except Exception as e:
                print(f"  ⚠ Error processing text at index {idx}: {e}")
                text_embeddings.append(np.zeros(MODEL2A_CONFIG["embedding_dim"]))
            
            # Progress update
            if (idx + 1) % 20 == 0:
                print(f"  → Processed {idx + 1}/{total} utterances ({(idx+1)/total*100:.1f}%)")
        
        # Add to dataframe
        self.prepared_data["text_embedding"] = text_embeddings
        
        print(f"\n✓ Text embeddings extracted")
        print(f"  Total: {len(text_embeddings)} x {MODEL2A_CONFIG['embedding_dim']} dims")
        
        # Save separately
        save_pickle({
            "embeddings": text_embeddings,
            "session_ids": self.prepared_data["session_id"].tolist(),
            "turn_ids": self.prepared_data["turn_id"].tolist()
        }, OUTPUTS_DIR / "text_embeddings.pkl")
    
    def extract_sentiment_features(self):
        """Extract sentiment features using VADER"""
        print_step(4, "Extracting Sentiment Features")
        
        if self.sentiment_analyzer is None:
            print("  ⚠ Sentiment analyzer not available, skipping...")
            self.prepared_data["sentiment_neg"] = 0.0
            self.prepared_data["sentiment_neu"] = 0.5
            self.prepared_data["sentiment_pos"] = 0.5
            self.prepared_data["sentiment_compound"] = 0.0
            return
        
        sentiments = []
        
        for idx, row in self.prepared_data.iterrows():
            text = str(row.get("text", ""))
            
            if not text or pd.isna(text):
                sentiments.append({
                    "neg": 0.0,
                    "neu": 0.5,
                    "pos": 0.5,
                    "compound": 0.0
                })
                continue
            
            try:
                scores = self.sentiment_analyzer.polarity_scores(text)
                sentiments.append(scores)
            except:
                sentiments.append({
                    "neg": 0.0,
                    "neu": 0.5,
                    "pos": 0.5,
                    "compound": 0.0
                })
        
        # Add to dataframe
        self.prepared_data["sentiment_neg"] = [s["neg"] for s in sentiments]
        self.prepared_data["sentiment_neu"] = [s["neu"] for s in sentiments]
        self.prepared_data["sentiment_pos"] = [s["pos"] for s in sentiments]
        self.prepared_data["sentiment_compound"] = [s["compound"] for s in sentiments]
        
        print(f"✓ Sentiment features extracted")
        print(f"  Average compound sentiment: {self.prepared_data['sentiment_compound'].mean():.3f}")
        
        # Save separately
        save_pickle({
            "sentiments": sentiments,
            "session_ids": self.prepared_data["session_id"].tolist()
        }, OUTPUTS_DIR / "sentiment_features.pkl")
    
    def engineer_visual_features(self):
        """Engineer visual features from AUs, pose, gaze"""
        print_step(5, "Engineering Visual Features")
        
        # Get visual feature columns
        au_cols = [c for c in self.prepared_data.columns if c.startswith("au_")]
        pose_cols = [c for c in self.prepared_data.columns if c.startswith("pose_")]
        gaze_cols = [c for c in self.prepared_data.columns if c.startswith("gaze_")]
        
        print(f"  Found features:")
        print(f"    - Action Units: {len(au_cols)}")
        print(f"    - Head Pose: {len(pose_cols)}")
        print(f"    - Gaze: {len(gaze_cols)}")
        
        # Fill NaN values
        for col in au_cols + pose_cols + gaze_cols:
            if col in self.prepared_data.columns:
                self.prepared_data[col] = self.prepared_data[col].fillna(0)
        
        # === ACTION UNIT FEATURES ===
        if au_cols:
            # Mean AU activation
            self.prepared_data["au_mean"] = self.prepared_data[au_cols].mean(axis=1)
            self.prepared_data["au_std"] = self.prepared_data[au_cols].std(axis=1)
            self.prepared_data["au_max"] = self.prepared_data[au_cols].max(axis=1)
            self.prepared_data["au_min"] = self.prepared_data[au_cols].min(axis=1)
            
            # Depression-related AUs (AU01, AU04, AU15)
            depression_au_cols = [c for c in au_cols if any(x in c for x in ["AU01", "AU04", "AU15"])]
            if depression_au_cols:
                self.prepared_data["au_depression_score"] = self.prepared_data[depression_au_cols].mean(axis=1)
            
            # Smile AUs (AU06, AU12)
            smile_au_cols = [c for c in au_cols if any(x in c for x in ["AU06", "AU12"])]
            if smile_au_cols:
                self.prepared_data["au_smile_score"] = self.prepared_data[smile_au_cols].mean(axis=1)
            
            print(f"  ✓ AU features: {len(au_cols)} → 6 aggregated features")
        
        # === HEAD POSE FEATURES ===
        if pose_cols:
            # Mean pose
            self.prepared_data["pose_mean"] = self.prepared_data[pose_cols].mean(axis=1)
            self.prepared_data["pose_std"] = self.prepared_data[pose_cols].std(axis=1)
            self.prepared_data["pose_range"] = self.prepared_data[pose_cols].max(axis=1) - self.prepared_data[pose_cols].min(axis=1)
            
            print(f"  ✓ Pose features: {len(pose_cols)} → 3 aggregated features")
        
        # === GAZE FEATURES ===
        if gaze_cols:
            # Mean gaze
            self.prepared_data["gaze_mean"] = self.prepared_data[gaze_cols].mean(axis=1)
            self.prepared_data["gaze_std"] = self.prepared_data[gaze_cols].std(axis=1)
            
            # Gaze avoidance (negative y values)
            y_gaze_cols = [c for c in gaze_cols if "_y_" in c or c.endswith("_y_0") or c.endswith("_y_1")]
            if y_gaze_cols:
                self.prepared_data["gaze_downward"] = (self.prepared_data[y_gaze_cols] < 0).sum(axis=1) / len(y_gaze_cols)
            
            print(f"  ✓ Gaze features: {len(gaze_cols)} → 3 aggregated features")
        
        # Save separately
        visual_features = {
            "au_cols": au_cols,
            "pose_cols": pose_cols,
            "gaze_cols": gaze_cols,
            "session_ids": self.prepared_data["session_id"].tolist()
        }
        save_pickle(visual_features, OUTPUTS_DIR / "visual_features.pkl")
        
        print(f"\n✓ Visual feature engineering complete")
    
    def create_session_aggregates(self):
        """Create session-level aggregated features"""
        print_step(6, "Creating Session-Level Aggregates")
        
        # Get all feature columns
        feature_cols = []
        
        # Text embedding (will average across turns)
        # Sentiment features
        feature_cols.extend(["sentiment_neg", "sentiment_neu", "sentiment_pos", "sentiment_compound"])
        
        # Visual aggregates
        agg_cols = [c for c in self.prepared_data.columns if any(
            c.startswith(x) for x in ["au_", "pose_", "gaze_"]
        )]
        feature_cols.extend(agg_cols)
        
        print(f"  Aggregating {len(feature_cols)} features per session")
        
        # Create session-level dataframe
        session_data = []
        
        for session_id in self.prepared_data["session_id"].unique():
            session_subset = self.prepared_data[self.prepared_data["session_id"] == session_id]
            
            # Text embeddings (mean across all turns)
            text_embs = np.stack(session_subset["text_embedding"].values)
            text_emb_mean = text_embs.mean(axis=0)
            
            # Other features (mean)
            feature_means = session_subset[feature_cols].mean().to_dict()
            
            # PHQ-8 score (same for all turns in session)
            phq8_score = session_subset["phq8_score"].iloc[0] if "phq8_score" in session_subset.columns else None
            
            session_data.append({
                "session_id": session_id,
                "text_embedding_mean": text_emb_mean,
                "num_turns": len(session_subset),
                "phq8_score": phq8_score,
                **feature_means
            })
        
        df_sessions = pd.DataFrame(session_data)
        
        print(f"✓ Session aggregates created")
        print(f"  Total sessions: {len(df_sessions)}")
        print(f"  Features per session: {len(feature_cols) + 1} (+ text embedding)")
        
        # Save
        save_pickle(df_sessions, OUTPUTS_DIR / "session_aggregates.pkl")
        
        return df_sessions
    
    def save_engineered_features(self):
        """Save complete engineered features"""
        print_step(7, "Saving Engineered Features")
        
        # Save full engineered data
        save_pickle(self.prepared_data, OUTPUTS_DIR / "engineered_features.pkl")
        
        # Also save as CSV for inspection
        csv_path = OUTPUTS_DIR / "engineered_features.csv"
        
        # Prepare CSV (convert embeddings to string)
        df_csv = self.prepared_data.copy()
        df_csv["text_embedding"] = df_csv["text_embedding"].apply(lambda x: f"[{len(x)}-dim vector]")
        df_csv.to_csv(csv_path, index=False)
        
        print(f"✓ Also saved as CSV: {csv_path}")
        
        # Summary statistics
        print(f"\n{'─' * 70}")
        print("FEATURE ENGINEERING SUMMARY")
        print(f"{'─' * 70}")
        print(f"Total utterances: {len(self.prepared_data)}")
        print(f"Total sessions: {self.prepared_data['session_id'].nunique()}")
        print(f"Total features: {len(self.prepared_data.columns)}")
        print(f"\nFeature categories:")
        print(f"  - Text embeddings: 768-dim (DistilBERT)")
        print(f"  - Sentiment: 4 features")
        print(f"  - Action Units: {len([c for c in self.prepared_data.columns if c.startswith('au_')])}")
        print(f"  - Head Pose: {len([c for c in self.prepared_data.columns if c.startswith('pose_')])}")
        print(f"  - Gaze: {len([c for c in self.prepared_data.columns if c.startswith('gaze_')])}")
        print(f"{'─' * 70}")
        
        print(f"\n✓ All features saved successfully!")
        print(f"  → Next step: python step3_model1_dialogue.py")

# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def test_feature_quality():
    """Test the quality of extracted features"""
    print_section("TESTING FEATURE QUALITY")
    
    # Load engineered features
    df = load_pickle(OUTPUTS_DIR / "engineered_features.pkl")
    
    print("1. Text Embedding Quality:")
    sample_emb = df["text_embedding"].iloc[0]
    print(f"   Dimension: {len(sample_emb)}")
    print(f"   Range: [{sample_emb.min():.3f}, {sample_emb.max():.3f}]")
    print(f"   Mean: {sample_emb.mean():.3f}")
    
    print("\n2. Sentiment Distribution:")
    print(f"   Negative: {df['sentiment_neg'].mean():.3f} ± {df['sentiment_neg'].std():.3f}")
    print(f"   Neutral: {df['sentiment_neu'].mean():.3f} ± {df['sentiment_neu'].std():.3f}")
    print(f"   Positive: {df['sentiment_pos'].mean():.3f} ± {df['sentiment_pos'].std():.3f}")
    print(f"   Compound: {df['sentiment_compound'].mean():.3f} ± {df['sentiment_compound'].std():.3f}")
    
    print("\n3. Visual Features:")
    if "au_mean" in df.columns:
        print(f"   AU mean: {df['au_mean'].mean():.3f} ± {df['au_mean'].std():.3f}")
    if "pose_mean" in df.columns:
        print(f"   Pose mean: {df['pose_mean'].mean():.3f} ± {df['pose_mean'].std():.3f}")
    if "gaze_mean" in df.columns:
        print(f"   Gaze mean: {df['gaze_mean'].mean():.3f} ± {df['gaze_mean'].std():.3f}")
    
    print("\n✓ Feature quality test complete")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution"""
    
    # Initialize
    fe = FeatureEngineering()
    
    # Step 1: Load prepared data
    if not fe.load_prepared_data():
        return
    
    # Step 2: Initialize text models
    if not fe.initialize_text_models():
        print("\n⚠ Text models not available. Some features will be missing.")
        # Continue anyway with available features
    
    # Step 3: Extract text embeddings
    if fe.tokenizer and fe.text_model:
        fe.extract_text_embeddings()
    else:
        print("\n⚠ Skipping text embeddings (DistilBERT not available)")
        # Add dummy embeddings
        fe.prepared_data["text_embedding"] = [np.zeros(768) for _ in range(len(fe.prepared_data))]
    
    # Step 4: Extract sentiment
    fe.extract_sentiment_features()
    
    # Step 5: Engineer visual features
    fe.engineer_visual_features()
    
    # Step 6: Create session aggregates
    fe.create_session_aggregates()
    
    # Step 7: Save
    fe.save_engineered_features()
    
    # Test
    test_feature_quality()
    
    print_section("FEATURE ENGINEERING COMPLETE")

if __name__ == "__main__":
    main()

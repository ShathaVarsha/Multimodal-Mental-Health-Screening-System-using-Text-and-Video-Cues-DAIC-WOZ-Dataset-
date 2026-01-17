"""
=================================================================
STEP 1: DATA PREPARATION
=================================================================
Extract and prepare data from all sessions (train: 303,304,305,310,312,313,315,316,317; test: 300,301)

This script:
1. Loads transcript data (text conversations)
2. Loads visual features (AUs, pose, gaze)
3. Loads PHQ-8 labels
4. Synchronizes timestamps between text and visual data
5. Saves prepared dataset for next steps

Inputs:
    - data/303,304,305,310,312,313,315,316,317/ (raw train session files)
    - data/300,301/ (raw test session files)
    - train_split_Depression_AVEC2017.csv
    - test_split_Depression_AVEC2017.csv

Outputs:
  - outputs/prepared_data.pkl (synchronized multimodal dataset)
  - outputs/session_statistics.json (data quality report)
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

# =============================================================================
# MAIN DATA PREPARATION CLASS
# =============================================================================

class DataPreparation:
    """Handles all data loading and preparation"""
    
    def __init__(self):
        """Initialize data preparation"""
        self.logger = setup_logging(LOG_FILE, LOG_LEVEL)
        print_section("STEP 1: DATA PREPARATION")
        
        self.sessions_data = {}
        self.phq8_labels = None
        self.prepared_dataset = []
        
    def load_phq8_labels(self):
        """Load PHQ-8 labels for all sessions"""
        print_step(1, "Loading PHQ-8 Labels")
        
        all_labels = []
        

        
        # Sessions 303, 304, 305 - Train split
        if TRAIN_SPLIT_FILE.exists():
            df_train = pd.read_csv(TRAIN_SPLIT_FILE)
            print(f"✓ Train split loaded: {len(df_train)} rows")
            print(f"  Columns: {list(df_train.columns)}")
            
            # Find ID column
            id_col = None
            for col in df_train.columns:
                if any(x in col.lower() for x in ['participant', 'session', 'id']):
                    id_col = col
                    break
            
            if id_col:
                for session_id in ["303", "304", "305"]:
                    session_data = df_train[df_train[id_col].astype(str) == session_id]
                    if len(session_data) > 0:
                        session_data = session_data.copy()
                        session_data["session_id"] = session_id
                        all_labels.append(session_data)
                        print(f"  ✓ Session {session_id} PHQ-8 loaded")
                    else:
                        print(f"  ⚠ Session {session_id} not found in train split")
            else:
                print(f"  ⚠ Could not find ID column in train split")
        else:
            print(f"❌ Train split file not found: {TRAIN_SPLIT_FILE}")
        
        # Sessions 300, 301 - Test split (NO PHQ VALUES)
        if TEST_SPLIT_FILE.exists():
            print(f"ℹ Test split exists but has no PHQ-8 values (sessions 300, 301)")
        
        # Combine all labels
        if all_labels:
            self.phq8_labels = pd.concat(all_labels, ignore_index=True)
            print(f"\n✓ Total sessions with PHQ-8 labels: {len(self.phq8_labels)}")
            print(f"  Sessions: {sorted(self.phq8_labels['session_id'].unique())}")
            
            # Show PHQ-8 columns
            phq_cols = [col for col in self.phq8_labels.columns if 'PHQ' in col.upper()]
            print(f"  PHQ-8 columns: {phq_cols}")
            
            return True
        else:
            print("\n❌ No PHQ-8 labels loaded")
            return False
    
    def load_session_transcript(self, session_id: str) -> pd.DataFrame:
        """
        Load and clean transcript for a session
        
        Args:
            session_id: Session identifier (e.g., "302")
            
        Returns:
            Cleaned transcript dataframe
        """
        session_path = DATA_DIR / session_id
        transcript_file = session_path / f"{session_id}_TRANSCRIPT.csv"
        
        if not transcript_file.exists():
            print(f"  ❌ Transcript not found: {transcript_file}")
            return None
        
        try:
            # Try tab-separated first, then comma
            try:
                df = pd.read_csv(transcript_file, sep="\t")
            except:
                df = pd.read_csv(transcript_file)
            
            # Normalize column names
            df.columns = [c.strip().lower() for c in df.columns]
            
            # Find text column
            text_col = None
            for col in df.columns:
                if col in ["value", "utterance", "text"]:
                    text_col = col
                    break
            
            if text_col is None:
                print(f"  ⚠ No text column found in transcript")
                return None
            
            # Find speaker column
            if "speaker" in df.columns:
                # Separate participant utterances
                df_participant = df[df["speaker"].str.lower() == "participant"].copy()
                df_ellie = df[df["speaker"].str.lower() == "ellie"].copy()
            else:
                df_participant = df.copy()
                df_ellie = pd.DataFrame()
            # Clean text
            df_participant[text_col] = df_participant[text_col].astype(str).str.strip()
            df_participant = df_participant[df_participant[text_col].str.len() > 0]
            
            # Add turn ID
            df_participant["turn_id"] = range(len(df_participant))
            
            # Store both participant and Ellie for dialogue training
            df_participant["is_participant"] = True
            if len(df_ellie) > 0:
                df_ellie[text_col] = df_ellie[text_col].astype(str).str.strip()
                df_ellie["is_participant"] = False
                df_ellie["turn_id"] = range(len(df_ellie))
            
            print(f"  ✓ Transcript: {len(df_participant)} participant, {len(df_ellie)} Ellie utterances")
            
            return {
                "participant": df_participant,
                "ellie": df_ellie,
                "combined": pd.concat([df_participant, df_ellie], ignore_index=True).sort_values("start_time") if len(df_ellie) > 0 else df_participant
            }
            
        except Exception as e:
            print(f"  ❌ Transcript error: {e}")
            return None
    
    def load_session_visual_features(self, session_id: str) -> Dict:
        """
        Load all visual features (AUs, pose, gaze) for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with visual feature dataframes
        """
        session_path = DATA_DIR / session_id
        visual_data = {}
        
        # Load Action Units
        au_file = session_path / f"{session_id}_CLNF_AUs.txt"
        if au_file.exists():
            try:
                df_au = pd.read_csv(au_file, sep=", ", engine="python")
                
                # Filter by confidence
                if "confidence" in df_au.columns:
                    df_au = df_au[df_au["confidence"] >= CONFIDENCE_THRESHOLD]
                if "success" in df_au.columns:
                    df_au = df_au[df_au["success"] == SUCCESS_THRESHOLD]
                
                visual_data["action_units"] = df_au
                print(f"  ✓ Action Units: {len(df_au)} frames")
            except Exception as e:
                print(f"  ❌ Action Units error: {e}")
        
        # Load Head Pose
        pose_file = session_path / f"{session_id}_CLNF_pose.txt"
        if pose_file.exists():
            try:
                df_pose = pd.read_csv(pose_file, sep=", ", engine="python")
                
                # Filter by confidence
                if "confidence" in df_pose.columns:
                    df_pose = df_pose[df_pose["confidence"] >= CONFIDENCE_THRESHOLD]
                if "success" in df_pose.columns:
                    df_pose = df_pose[df_pose["success"] == SUCCESS_THRESHOLD]
                
                # Sort by frame
                df_pose = df_pose.sort_values("frame").reset_index(drop=True)
                
                visual_data["pose"] = df_pose
                print(f"  ✓ Head Pose: {len(df_pose)} frames")
            except Exception as e:
                print(f"  ❌ Head Pose error: {e}")
        
        # Load Gaze
        gaze_file = session_path / f"{session_id}_CLNF_gaze.txt"
        if gaze_file.exists():
            try:
                df_gaze = pd.read_csv(gaze_file, sep=", ", engine="python")
                
                # Filter by confidence
                if "confidence" in df_gaze.columns:
                    df_gaze = df_gaze[df_gaze["confidence"] >= CONFIDENCE_THRESHOLD]
                if "success" in df_gaze.columns:
                    df_gaze = df_gaze[df_gaze["success"] == SUCCESS_THRESHOLD]
                
                # Sort by frame
                df_gaze = df_gaze.sort_values("frame").reset_index(drop=True)
                
                visual_data["gaze"] = df_gaze
                print(f"  ✓ Gaze: {len(df_gaze)} frames")
            except Exception as e:
                print(f"  ❌ Gaze error: {e}")
        
        return visual_data
    
    def synchronize_multimodal_data(self, session_id: str, 
                                     transcript: Dict, 
                                     visual: Dict) -> List[Dict]:
        """
        Synchronize transcript and visual features by timestamp
        
        Args:
            session_id: Session identifier
            transcript: Transcript data dictionary
            visual: Visual features dictionary
            
        Returns:
            List of synchronized data dictionaries
        """
        print(f"  → Synchronizing multimodal data...")
        
        synchronized = []
        
        # Use participant utterances
        df_participant = transcript["participant"]
        
        # Check if we have necessary columns
        if "start_time" not in df_participant.columns:
            print(f"  ⚠ No start_time column in transcript")
            return synchronized
        
        # Get text column
        text_col = None
        for col in df_participant.columns:
            if col in ["value", "utterance", "text"]:
                text_col = col
                break
        
        if text_col is None:
            print(f"  ⚠ No text column found")
            return synchronized
        
        # For each participant utterance, find corresponding visual features
        for idx, row in df_participant.iterrows():
            t_start = row["start_time"]
            t_end = row.get("stop_time", t_start + 5)  # Default 5 second window
            
            # Initialize synchronized row
            sync_row = {
                "session_id": session_id,
                "turn_id": row["turn_id"],
                "text": row[text_col],
                "start_time": t_start,
                "stop_time": t_end
            }
            
            # Extract Action Units in this time window
            if "action_units" in visual and len(visual["action_units"]) > 0:
                df_au = visual["action_units"]
                if "timestamp" in df_au.columns:
                    mask = (df_au["timestamp"] >= t_start) & (df_au["timestamp"] <= t_end)
                    au_segment = df_au[mask]
                    
                    if len(au_segment) > 0:
                        # Get AU columns (binary activation)
                        au_cols = [c for c in au_segment.columns if c.endswith("_c")]
                        
                        # Compute mean activation for each AU
                        for au in au_cols:
                            sync_row[f"au_{au}"] = au_segment[au].mean()
            
            # Extract Head Pose in this time window
            if "pose" in visual and len(visual["pose"]) > 0:
                df_pose = visual["pose"]
                if "timestamp" in df_pose.columns:
                    mask = (df_pose["timestamp"] >= t_start) & (df_pose["timestamp"] <= t_end)
                    pose_segment = df_pose[mask]
                    
                    if len(pose_segment) > 0:
                        # Get pose columns
                        pose_cols = ["pose_Rx", "pose_Ry", "pose_Rz", "pose_Tx", "pose_Ty", "pose_Tz"]
                        actual_pose_cols = ["Rx", "Ry", "Rz", "Tx", "Ty", "Tz"]
                        
                        for orig, new in zip(actual_pose_cols, pose_cols):
                            if orig in pose_segment.columns:
                                sync_row[new] = pose_segment[orig].mean()
            
            # Extract Gaze in this time window
            if "gaze" in visual and len(visual["gaze"]) > 0:
                df_gaze = visual["gaze"]
                if "timestamp" in df_gaze.columns:
                    mask = (df_gaze["timestamp"] >= t_start) & (df_gaze["timestamp"] <= t_end)
                    gaze_segment = df_gaze[mask]
                    
                    if len(gaze_segment) > 0:
                        # Get gaze columns
                        gaze_cols = [f"gaze_{c}" for c in ["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"]]
                        actual_gaze_cols = ["x_0", "y_0", "z_0", "x_1", "y_1", "z_1"]
                        
                        for orig, new in zip(actual_gaze_cols, gaze_cols):
                            if orig in gaze_segment.columns:
                                sync_row[new] = gaze_segment[orig].mean()
            
            synchronized.append(sync_row)
        
        print(f"  ✓ Synchronized {len(synchronized)} turns")
        return synchronized
    
    def prepare_all_sessions(self):
        """Load and prepare data for all training sessions"""
        print_step(2, "Loading All Training Sessions")
        
        for session_id in TRAIN_SESSIONS:
            print(f"\n{'─' * 70}")
            print(f"Processing Session {session_id}")
            print(f"{'─' * 70}")
            
            # Load transcript
            transcript = self.load_session_transcript(session_id)
            if transcript is None:
                print(f"  ⚠ Skipping session {session_id} - no transcript")
                continue
            
            # Load visual features
            visual = self.load_session_visual_features(session_id)
            if not visual:
                print(f"  ⚠ No visual features for session {session_id}")
            
            # Synchronize
            synchronized = self.synchronize_multimodal_data(session_id, transcript, visual)
            
            # Store
            self.sessions_data[session_id] = {
                "transcript": transcript,
                "visual": visual,
                "synchronized": synchronized
            }
        
        print(f"\n✓ Loaded {len(self.sessions_data)} sessions successfully")
    
    def merge_with_labels(self):
        """Merge synchronized data with PHQ-8 labels"""
        print_step(3, "Merging with PHQ-8 Labels")
        
        if self.phq8_labels is None or len(self.phq8_labels) == 0:
            print("❌ No PHQ-8 labels available")
            return
        
        for session_id, session_data in self.sessions_data.items():
            # Get PHQ-8 label for this session
            session_label = self.phq8_labels[self.phq8_labels["session_id"] == session_id]
            
            if len(session_label) == 0:
                print(f"  ⚠ No PHQ-8 label for session {session_id}")
                continue
            
            session_label = session_label.iloc[0]
            
            # Find PHQ-8 score column
            phq_score_col = None
            for col in session_label.index:
                if 'phq' in col.lower() and 'binary' not in col.lower():
                    phq_score_col = col
                    break
            
            if phq_score_col:
                phq_score = session_label[phq_score_col]
                print(f"  ✓ Session {session_id}: PHQ-8 Score = {phq_score}")
                
                # Add PHQ-8 score to each synchronized turn
                for turn in session_data["synchronized"]:
                    turn["phq8_score"] = phq_score
                    
                    # Add binary depression label (>=10 indicates depression)
                    turn["depression_binary"] = 1 if phq_score >= 10 else 0
        
        print(f"\n✓ Labels merged successfully")
    
    def create_final_dataset(self):
        """Create final prepared dataset"""
        print_step(4, "Creating Final Dataset")
        
        # Flatten all synchronized turns from all sessions
        all_turns = []
        
        for session_id, session_data in self.sessions_data.items():
            all_turns.extend(session_data["synchronized"])
        
        # Convert to DataFrame
        df_final = pd.DataFrame(all_turns)
        
        print(f"✓ Final dataset shape: {df_final.shape}")
        print(f"  Total turns: {len(df_final)}")
        print(f"  Sessions: {df_final['session_id'].nunique()}")
        print(f"  Features: {len(df_final.columns)}")
        
        # Show column categories
        au_cols = [c for c in df_final.columns if c.startswith("au_")]
        pose_cols = [c for c in df_final.columns if c.startswith("pose_")]
        gaze_cols = [c for c in df_final.columns if c.startswith("gaze_")]
        
        print(f"\n  Feature breakdown:")
        print(f"    - Text features: 1 (text)")
        print(f"    - Action Units: {len(au_cols)}")
        print(f"    - Head Pose: {len(pose_cols)}")
        print(f"    - Gaze: {len(gaze_cols)}")
        print(f"    - Labels: 2 (phq8_score, depression_binary)")
        
        # Show PHQ-8 distribution
        if "phq8_score" in df_final.columns:
            print(f"\n  PHQ-8 Score distribution:")
            print(df_final.groupby("session_id")["phq8_score"].first().describe())
        
        self.prepared_dataset = df_final
        
        return df_final
    
    def save_prepared_data(self):
        """Save prepared dataset"""
        print_step(5, "Saving Prepared Data")
        
        # Save as pickle
        save_pickle(self.prepared_dataset, OUTPUTS_DIR / "prepared_data.pkl")
        
        # Save as CSV for inspection
        csv_path = OUTPUTS_DIR / "prepared_data.csv"
        self.prepared_dataset.to_csv(csv_path, index=False)
        print(f"✓ Also saved as CSV: {csv_path}")
        
        # Save session statistics
        stats = {
            "total_sessions": len(self.sessions_data),
            "total_turns": len(self.prepared_dataset),
            "session_ids": list(self.sessions_data.keys()),
            "features": {
                "text": 1,
                "action_units": len([c for c in self.prepared_dataset.columns if c.startswith("au_")]),
                "pose": len([c for c in self.prepared_dataset.columns if c.startswith("pose_")]),
                "gaze": len([c for c in self.prepared_dataset.columns if c.startswith("gaze_")])
            }
        }
        
        save_json(stats, OUTPUTS_DIR / "session_statistics.json")
        
        print(f"\n✓ All data preparation complete!")
        print(f"  → Next step: python feature_engineering_2.py")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution"""
    
    # Initialize
    prep = DataPreparation()
    
    # Step 1: Load PHQ-8 labels
    if not prep.load_phq8_labels():
        print("\n❌ Failed to load PHQ-8 labels. Please check your CSV files.")
        return
    
    # Step 2: Load all sessions
    prep.prepare_all_sessions()
    
    if len(prep.sessions_data) == 0:
        print("\n❌ No sessions loaded. Please check your data directory.")
        return
    
    # Step 3: Merge with labels
    prep.merge_with_labels()
    
    # Step 4: Create final dataset
    prep.create_final_dataset()
    
    # Step 5: Save
    prep.save_prepared_data()
    
    print_section("DATA PREPARATION COMPLETE")

if __name__ == "__main__":
    main()

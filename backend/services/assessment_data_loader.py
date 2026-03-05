import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


class AssessmentDataLoader:
    """Loads assessment question banks and participant divisions from data files."""

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.data_dir = self.root_dir / "data"
        self.question_bank_file = self.data_dir / "question_bank.json"

    def get_questions(self, section: str, scenario: str = None) -> List[Dict]:
        """Load questions from data/question_bank.json by section name.
        
        For scenario-specific follow-ups, pass section='phq9_scenario_followups' and scenario='postpartum'
        """
        if not self.question_bank_file.exists():
            return []

        try:
            with open(self.question_bank_file, "r", encoding="utf-8") as file:
                payload = json.load(file)
            
            # Handle scenario-specific follow-ups
            if section == 'phq9_scenario_followups' and scenario:
                followups_dict = payload.get('phq9_scenario_followups', {})
                questions = followups_dict.get(scenario, [])
            else:
                questions = payload.get(section, [])
            
            return questions if isinstance(questions, list) else []
        except Exception:
            return []

    def get_divisions(self, include_ids: bool = False) -> Dict:
        """Load train/dev/test/full participant divisions from split CSV files."""
        split_files = {
            "train": self.root_dir / "train_split_Depression_AVEC2017.csv",
            "dev": self.root_dir / "test_split_Depression_AVEC2017.csv",
            "test": self.root_dir / "dev_split_Depression_AVEC2017.csv",
            "full_test": self.root_dir / "full_test_split.csv"
        }

        result = {}
        for split_name, split_path in split_files.items():
            participants = self._read_split_participants(split_path)
            labels = [p.get("label") for p in participants if p.get("label") is not None]
            depressed_count = sum(1 for value in labels if str(value) == "1")
            control_count = sum(1 for value in labels if str(value) == "0")

            split_data = {
                "split": split_name,
                "source_file": split_path.name,
                "total_participants": len(participants),
                "depressed_count": depressed_count,
                "control_count": control_count
            }

            if include_ids:
                split_data["participant_ids"] = [p["participant_id"] for p in participants]

            result[split_name] = split_data

        return result

    def is_participant_assessable(self, participant_id: int, split: str = "full_test") -> bool:
        """Check whether a participant exists in a given split."""
        divisions = self.get_divisions(include_ids=True)
        split_data = divisions.get(split, {})
        ids = split_data.get("participant_ids", [])
        return int(participant_id) in ids if ids else False

    @staticmethod
    def _read_split_participants(split_path: Path) -> List[Dict]:
        if not split_path.exists():
            return []

        records: List[Dict] = []
        with open(split_path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                participant_id = (
                    row.get("participant_id")
                    or row.get("Participant_ID")
                    or row.get("ParticipantID")
                    or row.get("id")
                )
                if participant_id is None:
                    continue
                try:
                    participant_id = int(float(participant_id))
                except Exception:
                    continue

                label = row.get("label")
                records.append({"participant_id": participant_id, "label": label})

        return records

"""
PTSD (PCL-5) Report Generator
Generates professional PTSD assessment reports from chat/text data.
- Loads chat/text data from data/PCL and data/PCL5
- Calculates PCL-5 score and severity
- Generates a structured report (PDF-ready)
- Integrates crisis detection and severity-based output
"""
import os
import pandas as pd
from datetime import datetime
from fpdf import FPDF

# PCL-5 severity cutoffs (Weathers et al., 2013)
PCL5_CUTOFFS = [20, 30, 33, 46, 80]
PCL5_LABELS = ["Minimal", "Mild", "Probable PTSD", "Moderate", "Severe"]

CRISIS_KEYWORDS = [
    "kill myself", "want to die", "end it", "suicide", "overdose", "hang myself", "cut myself", "already planned", "I know how", "I've decided", "I have pills", "nothing can stop me", "it's over", "I tried last week", "I tried yesterday", "I cut myself", "I overdosed", "never get better", "no hope", "nobody cares", "no reason to exist"
]
PROTECTIVE_KEYWORDS = [
    "my kids", "my family", "reasons to live", "my faith", "want to see", "people would be hurt"
]

CRISIS_RESOURCES = {
    "988 Suicide & Crisis Lifeline": "988",
    "Crisis Text Line": "Text HOME to 741741",
    "Veterans Crisis": "988 then press 1",
    "911 Emergency": "911"
}

def detect_crisis(text):
    score = 0
    text = text.lower()
    for word in CRISIS_KEYWORDS:
        if word in text:
            score += 1
    for word in PROTECTIVE_KEYWORDS:
        if word in text:
            score -= 1
    return max(0, min(score, 10))

def pcl5_severity(score):
    if score <= 20:
        return "Minimal"
    elif score <= 30:
        return "Mild"
    elif score <= 33:
        return "Probable PTSD"
    elif score <= 46:
        return "Moderate"
    else:
        return "Severe"

class PTSDReportPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'PTSD (PCL-5) ASSESSMENT REPORT', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.cell(0, 8, 'For Professional Clinical Use', 0, 1, 'C')
        self.ln(4)
    def section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, title, 0, 1)
        self.ln(2)
    def section_body(self, text):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 6, text)
        self.ln(2)
    def add_section(self, title, body):
        self.section_title(title)
        self.section_body(body)
    def generate(self, filename):
        self.output(filename)

def generate_ptsd_report(session_id, chat_df, pcl5_scores, transcript, output_path, name='', occupation=''):
    total_score = sum(pcl5_scores)
    severity = pcl5_severity(total_score)
    crisis_score = detect_crisis(" ".join(transcript))
    pdf = PTSDReportPDF()
    pdf.add_page()
    # Personalized header
    meta = f"Session ID: {session_id}\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    if name:
        meta += f"\nName: {name}"
    if occupation:
        meta += f"\nOccupation: {occupation}"
    pdf.add_section('ASSESSMENT METADATA', meta)
    pdf.add_section('CLINICAL SUMMARY', f"PCL-5 Score: {total_score}/80\nSeverity: {severity}\nCrisis Score: {crisis_score}/10")
    pdf.add_section('SYMPTOM BREAKDOWN', "\n".join([f"Item {i+1}: {score}/4" for i, score in enumerate(pcl5_scores)]))
    pdf.add_section('TRANSCRIPT', "\n".join(transcript[:20]) + ("\n..." if len(transcript) > 20 else ""))
    if crisis_score >= 8:
        pdf.add_section('CRISIS RESOURCES', "\n".join([f"{k}: {v}" for k, v in CRISIS_RESOURCES.items()]))
    pdf.generate(output_path)
    return output_path

# Example usage:
# chat_df = pd.read_csv('data/PCL/session1.csv')
# pcl5_scores = [int(x) for x in chat_df['score']]
# transcript = list(chat_df['text'])
# generate_ptsd_report('session1', chat_df, pcl5_scores, transcript, 'ptsd_report_session1.pdf')

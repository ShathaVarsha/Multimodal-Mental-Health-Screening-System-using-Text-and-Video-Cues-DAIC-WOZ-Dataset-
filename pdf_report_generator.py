# PDF Report Generator for Mental Health Assessment
# Uses fpdf for simplicity; can be swapped for reportlab if needed
from fpdf import FPDF
from datetime import datetime

class AssessmentPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'MENTAL HEALTH SCREENING REPORT', 0, 1, 'C')
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

    def add_video_description(self, video_desc):
        self.section_title('VIDEO DESCRIPTION (Depression)')
        self.section_body(video_desc)

    def add_section(self, title, body):
        self.section_title(title)
        self.section_body(body)

    def add_transcript(self, transcript):
        self.section_title('CONVERSATION TRANSCRIPT & ANALYSIS')
        self.section_body(transcript)

    def add_metadata(self, metadata):
        self.section_title('ASSESSMENT METADATA')
        self.section_body(metadata)

    def add_summary(self, summary):
        self.section_title('CLINICAL SUMMARY')
        self.section_body(summary)

    def add_symptom_analysis(self, analysis):
        self.section_title('DETAILED SYMPTOM ANALYSIS')
        self.section_body(analysis)

    def add_risk_assessment(self, risk):
        self.section_title('RISK ASSESSMENT & SAFETY')
        self.section_body(risk)

    def add_recommendations(self, recs):
        self.section_title('EVIDENCE-BASED RECOMMENDATIONS')
        self.section_body(recs)

    def add_referral(self, referral):
        self.section_title('REFERRAL & COORDINATION')
        self.section_body(referral)

    def add_limitations(self, limitations):
        self.section_title('LIMITATIONS & DISCLAIMERS')
        self.section_body(limitations)

    def add_clinician_notes(self, notes):
        self.section_title('CLINICIAN NOTES & ACTION ITEMS')
        self.section_body(notes)

    def add_validity(self, validity):
        self.section_title('ASSESSMENT VALIDITY & SOURCES')
        self.section_body(validity)

    def add_patient_summary(self, summary):
        self.section_title('PATIENT SUMMARY (For Patient Copy)')
        self.section_body(summary)

    def generate(self, filename):
        self.output(filename)

# Example usage:
# pdf = AssessmentPDF()
# pdf.add_metadata('Date: ...')
# pdf.add_video_description('Video analysis: ...')
# pdf.generate('Assessment_Report.pdf')

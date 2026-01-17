"""
ptsd_agent.py
A detailed, story-based PTSD agent using validated PCL-5 questions, trauma prompts, and scoring logic.
"""
import datetime
from fpdf import FPDF

# PCL-5 questions and descriptions
PCL5_QUESTIONS = [
    "Repeated, disturbing, and unwanted memories of the stressful experience?",
    "Repeated, disturbing dreams of the stressful experience?",
    "Suddenly feeling or acting as if the stressful experience were actually happening again (as if you were actually back there reliving it)?",
    "Feeling very upset when something reminded you of the stressful experience?",
    "Having strong physical reactions when something reminded you of the stressful experience (for example, heart pounding, trouble breathing, sweating)?",
    "Avoiding memories, thoughts, or feelings related to the stressful experience?",
    "Avoiding external reminders of the stressful experience (for example, people, places, conversations, activities, objects, or situations)?",
    "Trouble remembering important parts of the stressful experience?",
    "Having strong negative beliefs about yourself, other people, or the world (for example, having thoughts such as: I am bad, there is something seriously wrong with me, no one can be trusted, the world is completely dangerous)?",
    "Blaming yourself or someone else for the stressful experience or what happened after it?",
    "Having strong negative feelings such as fear, horror, anger, guilt, or shame?",
    "Loss of interest in activities that you used to enjoy?",
    "Feeling distant or cut off from other people?",
    "Trouble experiencing positive feelings (for example, being unable to feel happiness or have loving feelings for people close to you)?",
    "Irritable behavior, angry outbursts, or acting aggressively?",
    "Taking too many risks or doing things that could cause you harm?",
    "Being “superalert” or watchful or on guard?",
    "Feeling jumpy or easily startled?",
    "Having difficulty concentrating?",
    "Trouble falling or staying asleep?"
]

PCL5_OPTIONS = [
    "0 - Not at all",
    "1 - A little bit",
    "2 - Moderately",
    "3 - Quite a bit",
    "4 - Extremely"
]

TRAUMA_PROMPT = (
    "This questionnaire asks about problems you may have had after a very stressful experience involving "
    "actual or threatened death, serious injury, or sexual violence. It could be something that happened to you directly, "
    "something you witnessed, or something you learned happened to a close family member or close friend. "
    "Some examples are a serious accident; fire; disaster such as a hurricane, tornado, or earthquake; physical or sexual attack or abuse; war; homicide; or suicide.\n"
    "First, please briefly describe the worst event that currently bothers you the most (optional): "
)

# Empathetic, story-based introduction
INTRO = (
    "Welcome. I'm here to help you talk about difficult experiences in a safe, supportive way. "
    "We'll go through some questions about how you've been feeling after a stressful event. "
    "You can skip any question you don't want to answer."
)

class PTSDPDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'PTSD Screening Report (PCL-5)', ln=True, align='C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def run_ptsd_agent():
    print(INTRO)
    trauma = input(TRAUMA_PROMPT)
    responses = []
    print("\nFor each of the following, please answer how much you've been bothered in the past month:")
    for idx, q in enumerate(PCL5_QUESTIONS, 1):
        print(f"\n{idx}. {q}")
        for opt in PCL5_OPTIONS:
            print(opt)
        while True:
            try:
                ans = int(input("Your answer (0-4): "))
                if 0 <= ans <= 4:
                    responses.append(ans)
                    break
                else:
                    print("Please enter a number from 0 to 4.")
            except ValueError:
                print("Please enter a valid number.")
    total_score = sum(responses)
    print(f"\nYour total PCL-5 score: {total_score}")
    # Severity interpretation
    if total_score >= 33:
        severity = "High (PTSD likely, further assessment recommended)"
    elif total_score >= 21:
        severity = "Moderate (possible PTSD, monitor symptoms)"
    else:
        severity = "Low (unlikely PTSD, but monitor if symptoms persist)"
    print(f"Severity: {severity}")
    # Generate PDF report
    report = PTSDPDFReport()
    report.add_page()
    report.set_font('Arial', '', 12)
    report.cell(0, 10, f'Date: {datetime.date.today()}', ln=True)
    report.multi_cell(0, 10, f"Trauma description: {trauma if trauma else 'Not provided'}")
    report.ln(5)
    report.set_font('Arial', 'B', 12)
    report.cell(0, 10, 'PCL-5 Responses:', ln=True)
    report.set_font('Arial', '', 12)
    for idx, (q, a) in enumerate(zip(PCL5_QUESTIONS, responses), 1):
        report.multi_cell(0, 8, f"{idx}. {q}\n   Response: {PCL5_OPTIONS[a]}")
    report.ln(5)
    report.set_font('Arial', 'B', 12)
    report.cell(0, 10, f'Total PCL-5 Score: {total_score}', ln=True)
    report.cell(0, 10, f'Severity: {severity}', ln=True)
    report.output('ptsd_report.pdf')
    print("\nA detailed PDF report has been generated: ptsd_report.pdf")

if __name__ == "__main__":
    run_ptsd_agent()

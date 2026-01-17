"""
ptsd_chat_agent.py
A full-featured, chat-only PTSD AI agent supporting both PCL (DSM-IV) and PCL-5 (DSM-5) with empathetic, story-based flow and PDF reporting.
"""
import datetime
from fpdf import FPDF

# PCL (DSM-IV, 17 items)
PCL_QUESTIONS = [
    "Repeated, disturbing memories, thoughts, or images of a stressful experience from the past?",
    "Repeated, disturbing dreams of a stressful experience from the past?",
    "Suddenly acting or feeling as if a stressful experience were happening again (as if you were reliving it)?",
    "Feeling very upset when something reminded you of a stressful experience from the past?",
    "Having physical reactions (e.g., heart pounding, trouble breathing, sweating) when something reminded you of a stressful experience from the past?",
    "Avoiding thinking about or talking about a stressful experience from the past or avoiding having feelings related to it?",
    "Avoiding activities or situations because they reminded you of a stressful experience from the past?",
    "Trouble remembering important parts of a stressful experience from the past?",
    "Loss of interest in activities that you used to enjoy?",
    "Feeling distant or cut off from other people?",
    "Feeling emotionally numb or being unable to have loving feelings for those close to you?",
    "Feeling as if your future will somehow be cut short?",
    "Trouble falling or staying asleep?",
    "Feeling irritable or having angry outbursts?",
    "Having difficulty concentrating?",
    "Being 'super-alert' or watchful or on guard?",
    "Feeling jumpy or easily startled?"
]

# PCL-5 (DSM-5, 20 items)
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

PCL_OPTIONS = [
    "1 - Not at all",
    "2 - A little bit",
    "3 - Moderately",
    "4 - Quite a bit",
    "5 - Extremely"
]

PCL5_OPTIONS = [
    "0 - Not at all",
    "1 - A little bit",
    "2 - Moderately",
    "3 - Quite a bit",
    "4 - Extremely"
]

INTRO = (
    "Welcome. I'm your AI assistant for PTSD screening. We'll talk about your experiences in a safe, supportive way. "
    "You can choose between two validated questionnaires: PCL (17 questions, DSM-IV) or PCL-5 (20 questions, DSM-5). "
    "All your answers are confidential. You can skip any question. Let's begin."
)

class PTSDChatPDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'PTSD Screening Report', ln=True, align='C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def ask_questions(questions, options):
    responses = []
    for idx, q in enumerate(questions, 1):
        print(f"\n{idx}. {q}")
        for opt in options:
            print(opt)
        while True:
            ans = input("Your answer (number, or 's' to skip): ").strip()
            if ans.lower() == 's':
                responses.append(None)
                break
            try:
                ans_int = int(ans)
                if 0 <= ans_int <= len(options)-1:
                    responses.append(ans_int)
                    break
                else:
                    print(f"Please enter a number from 0 to {len(options)-1}.")
            except ValueError:
                print("Please enter a valid number or 's' to skip.")
    return responses

def interpret_pcl_score(score):
    if score >= 44:
        return "High (PTSD likely, further assessment recommended)"
    elif score >= 30:
        return "Moderate (possible PTSD, monitor symptoms)"
    else:
        return "Low (unlikely PTSD, but monitor if symptoms persist)"

def interpret_pcl5_score(score):
    if score >= 33:
        return "High (PTSD likely, further assessment recommended)"
    elif score >= 21:
        return "Moderate (possible PTSD, monitor symptoms)"
    else:
        return "Low (unlikely PTSD, but monitor if symptoms persist)"

def run_ptsd_chat_agent():
    print(INTRO)
    print("\nWhich questionnaire would you like to use?")
    print("1. PCL (DSM-IV, 17 questions, 1-5 scale)")
    print("2. PCL-5 (DSM-5, 20 questions, 0-4 scale)")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == '1':
            version = 'PCL'
            questions = PCL_QUESTIONS
            options = PCL_OPTIONS
            break
        elif choice == '2':
            version = 'PCL-5'
            questions = PCL5_QUESTIONS
            options = PCL5_OPTIONS
            break
        else:
            print("Please enter 1 or 2.")
    trauma = input("\n(Optional) Briefly describe the most stressful event you want to focus on: ")
    print("\nFor each of the following, please answer how much you've been bothered in the past month:")
    responses = ask_questions(questions, options)
    # Calculate score (skip None)
    score = sum([r for r in responses if r is not None])
    if version == 'PCL':
        severity = interpret_pcl_score(score)
    else:
        severity = interpret_pcl5_score(score)
    print(f"\nYour total {version} score: {score}")
    print(f"Severity: {severity}")
    # Generate PDF report
    report = PTSDChatPDFReport()
    report.add_page()
    report.set_font('Arial', '', 12)
    report.cell(0, 10, f'Date: {datetime.date.today()}', ln=True)
    report.multi_cell(0, 10, f"Trauma description: {trauma if trauma else 'Not provided'}")
    report.ln(5)
    report.set_font('Arial', 'B', 12)
    report.cell(0, 10, f'{version} Responses:', ln=True)
    report.set_font('Arial', '', 12)
    for idx, (q, a) in enumerate(zip(questions, responses), 1):
        if a is not None:
            report.multi_cell(0, 8, f"{idx}. {q}\n   Response: {options[a]}")
        else:
            report.multi_cell(0, 8, f"{idx}. {q}\n   Response: Skipped")
    report.ln(5)
    report.set_font('Arial', 'B', 12)
    report.cell(0, 10, f'Total {version} Score: {score}', ln=True)
    report.cell(0, 10, f'Severity: {severity}', ln=True)
    report.output('ptsd_chat_report.pdf')
    print("\nA detailed PDF report has been generated: ptsd_chat_report.pdf")

if __name__ == "__main__":
    run_ptsd_chat_agent()

import pdfplumber
import os

# Paths to PCL and PCL5 folders
PCL_DIR = os.path.join('data', 'PCL')
PCL5_DIR = os.path.join('data', 'PCL5')

# List of PDF files to extract from
def list_pdfs(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.pdf')]

# Extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ''
            text += '\n'
    return text

# Main extraction logic
def extract_all_pcl_questions():
    all_text = {}
    for folder in [PCL_DIR, PCL5_DIR]:
        for pdf_file in list_pdfs(folder):
            print(f'Extracting from: {pdf_file}')
            text = extract_text_from_pdf(pdf_file)
            all_text[os.path.basename(pdf_file)] = text
    return all_text

if __name__ == '__main__':
    all_pcl_text = extract_all_pcl_questions()
    # Save to a text file for review
    with open('extracted_pcl_questions.txt', 'w', encoding='utf-8') as f:
        for fname, text in all_pcl_text.items():
            f.write(f'===== {fname} =====\n')
            f.write(text)
            f.write('\n\n')
    print('Extraction complete. See extracted_pcl_questions.txt.')

from pdfminer.high_level import extract_text

def pdf_to_text(file_path):
    return extract_text(file_path)

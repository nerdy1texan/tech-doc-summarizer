from transformers import T5Tokenizer, T5ForConditionalGeneration
from pdf_to_text import pdf_to_text  
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#model loading
tokenizer = T5Tokenizer.from_pretrained('./models/tokenizer/')
model = T5ForConditionalGeneration.from_pretrained('./models/model/').to(device)

def summarize_text(text, max_length=200):
    input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", truncation=True).to(device)
    summary_ids = model.generate(input_ids, num_beams=4, max_length=max_length, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main(pdf_file_path):
    extracted_text = pdf_to_text(pdf_file_path)
    summary = summarize_text(extracted_text)
    
    return summary

if __name__ == "__main__":
    #not useful if doing through UI
    pdf_file_path = "pdf\AI_summarization_paper.pdf"
    
    summary = main(pdf_file_path)
    print(summary)

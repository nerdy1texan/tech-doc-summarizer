import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from pdf_to_text import pdf_to_text #text xtraction func
from PIL import Image

st.set_page_config(page_title='Technical Report Summarizer', page_icon='📄')

#model loading + tokenizer
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(f'./models/{model_name}/tokenizer/')
    model = AutoModelForSeq2SeqLM.from_pretrained(f'./models/{model_name}/model/').to("cuda")
    return tokenizer, model

logo_path='AAI_lab_logo.png'
st.image(logo_path, width=100, use_column_width=True)
st.markdown("""<br><br><br><br><br><br>""", unsafe_allow_html=True)
st.title('Technical Report Summarization')

#mod selection
available_models = ['t5-11B','t5-large', 'mistral-7B', 'pegasus-x']  # List your model names here
selected_model = st.selectbox('Choose a model for summarization:', available_models)

#pdf uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    with open(f'pdf/{uploaded_file.name}', 'wb') as f:
        f.write(uploaded_file.getbuffer())
        
    st.success('File successfully uploaded!')

    #imported @ top
    extracted_text = pdf_to_text(f'pdf/{uploaded_file.name}')

    if st.button('Summarize'):
        #mod loading
        tokenizer, model = load_model(selected_model)

        #text -> tokens
        inputs = tokenizer.encode("summarize: " + extracted_text, return_tensors='pt', max_length=1024, truncation=True).to("cuda")
        summary_ids = model.generate(inputs, max_length=1024, min_length=512, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        st.write(summary)








st.markdown("""<br><br>""", unsafe_allow_html=True)


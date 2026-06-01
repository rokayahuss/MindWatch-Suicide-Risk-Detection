import streamlit as st
import torch
import joblib
import re
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

# Page Configuration
st.set_page_config(page_title="MindWatch Platform", page_icon="🧠", layout="centered")

# Cached function to load models once
@st.cache_resource
def load_models():
    vectorizer = joblib.load('best_vectorizer.pkl')
    ml_model = joblib.load('best_ml_model.pkl')
    bert_tokenizer = BertTokenizer.from_pretrained('./saved_bert_model')
    bert_model = BertForSequenceClassification.from_pretrained('./saved_bert_model')
    bert_model.eval()
    return vectorizer, ml_model, bert_tokenizer, bert_model

try:
    vectorizer, ml_model, bert_tokenizer, bert_model = load_models()
except Exception as e:
    st.error("Error: Make sure you have executed the training code and saved the models first!")

# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# UI Layout
st.title("🧠 MindWatch: Suicide Risk Detection Platform")
st.markdown("Analyze online posts, essays, or text sequences to detect potential indications of self-harm or suicide risks.")
st.markdown("---")

# Input Components
user_input = st.text_area("Input Text Content:", height=150, placeholder="Type or paste your text here...")
model_option = st.selectbox("Select Model Core:", ("Fine-tuned BERT Transformer", "Traditional ML Model (SVM / Naive Bayes)"))

# Inference Trigger
if st.button("Run MindWatch Analysis"):
    if user_input.strip() == "":
        st.warning("Please provide a valid text input before analyzing.")
    else:
        cleaned_text = clean_text(user_input)

        with st.spinner("Analyzing semantics and context..."):
            if model_option == "Traditional ML Model (SVM / Naive Bayes)":
                vec_text = vectorizer.transform([cleaned_text])
                prediction = ml_model.predict(vec_text)[0]
                confidence_info = "Classification based on the model's decision boundary."
            else:
                inputs = bert_tokenizer(cleaned_text, truncation=True, padding=True, max_length=128, return_tensors="pt")
                with torch.no_grad():
                    outputs = bert_model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).flatten().tolist()
                    prediction = np.argmax(probs)
                    confidence_info = f"Transformer Intent Confidence: {probs[prediction]*100:.2f}%"

        # Results Presentation
        st.subheader("Analysis Breakdown:")

        if prediction == 1:
            st.error("Flagged Notification: Suicide Intent Risk Detected.")
            st.metric(label="Status", value="High Risk Indicator")
        else:
            st.success("Analysis Clear: Non-Suicide Text Profile.")
            st.metric(label="Status", value="Low/No Risk Detected")

        st.info(f"ℹ**Metadata Details:** {confidence_info}")

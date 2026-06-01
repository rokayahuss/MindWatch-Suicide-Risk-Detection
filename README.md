# MindWatch: Suicide Risk Detection Platform

MindWatch is an advanced Natural Language Processing (NLP) platform designed to analyze text sequences and classify potential indications of self-harm, distress, or suicide risks. This project explores and compares the computational and architectural trade-offs between traditional Machine Learning models and modern Deep Learning Transformer architectures.

---

## Tech Stack & Library Frameworks
* **Programming Language:** Python
* **Deep Learning Framework:** PyTorch & Hugging Face (Transformers)
* **Traditional Machine Learning:** Scikit-Learn
* **Natural Language Processing:** NLTK (Tokenization, Lemmatization, Stopwords handling)
* **Data Processing & Visualization:** Pandas, NumPy, Regex, Seaborn, Matplotlib
* **UI Platform Deployment:** Streamlit

---

## Core Features
* **Rigorous Text Preprocessing:** Custom cleaning functions incorporating Regex for stripping URLs, non-ASCII characters, numbers, and excess whitespaces, followed by advanced tokenization and word lemmatization.
* **Dual-Architecture Classification:** * **Traditional Pipeline:** Implementation of Linear Support Vector Classification (`LinearSVC`) and Naive Bayes (`MultinomialNB`) paired with TF-IDF/Bag-of-Words vectorization.
  * **Deep Learning Pipeline:** Fine-tuned `bert-base-uncased` Sequence Classification model leveraging transfer learning.
* **Production-Ready Web Dashboard:** A clean, interactive frontend built entirely via `Streamlit` allowing end-users to select architectures and run instant inference on text datasets.

---

## Architectural Trade-offs & Insights Summary
During rigorous testing and validation, a fascinating machine learning paradigm was highlighted:
1. **Traditional ML (SVM):** Demonstrated highly robust performance when handling specific keyword triggers and localized semantic patterns due to extensive training over the complete global dataset weights.
2. **Transformer DL (BERT):** Exhibited deep semantic and contextual awareness of complex sentence structures and underlying emotional sentiments. However, it requires significantly scaled-up training samples and fine-tuning epochs to perfectly establish finer class boundaries.

---

## How to Run the Interface Locally
1. Clone this repository or download the source files.
2. Install the necessary packages:
   ```bash
   pip install streamlit torch transformers scikit-learn joblib

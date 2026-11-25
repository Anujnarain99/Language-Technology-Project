# Spam and Phishing Email/SMS Detection System
**Course:** ETMMTI303 - Language Technologies and Text Intelligence | K.R. Mangalam University

## Overview
An end-to-end NLP application comparing a classical statistical baseline (**TF-IDF + Multinomial Naive Bayes**) against a pretrained transformer (**BERT-Tiny fine-tuned for SMS spam**) via an interactive **Streamlit** dashboard.

## Dataset
- **Source:** UCI / PyCon SMS Spam Collection (spam.csv)
- **Size:** 5,574 raw records (5,169 unique records after deduplication)
- **Classes:** ham (legitimate, 0) and spam (spam/phishing, 1)

## Results Summary
- **Multinomial Naive Bayes (TF-IDF):** 98% overall accuracy (Spam Precision: 0.99, Recall: 0.87, F1: 0.93) on 1,032 test messages.
- **Pretrained BERT Transformer:** 99% overall accuracy (Spam Precision: 1.00, Recall: 0.90, F1: 0.95) on a 200-message test sample.

## How to Run
1. pip install -r requirements.txt
2. python train_and_eval.py
3. python -m streamlit run app.py

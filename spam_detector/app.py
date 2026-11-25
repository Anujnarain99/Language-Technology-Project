import re, joblib, nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import pipeline

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

st.set_page_config(page_title='Spam & Phishing Detector', layout='centered')
st.title('Spam & Phishing Message Detector')
st.caption('Anuj (2501940039), Ramesh Kumar (2501940086)')

@st.cache_resource
def load_models():
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    nb_model = joblib.load('nb_model.pkl')
    tf_clf = pipeline('text-classification', model='mrm8488/bert-tiny-finetuned-sms-spam-detection', truncation=True)
    return tfidf, nb_model, tf_clf

tfidf, nb_model, tf_clf = load_models()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' url ', text)
    text = re.sub(r'\d+', ' num ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join([lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words])

tab1, tab2, tab3 = st.tabs(['Single Message Check', 'Batch CSV Upload', 'Evaluation Charts'])

with tab1:
    user_input = st.text_area('Enter an email or SMS message:', height=120, placeholder='URGENT! Click http://verify-now.com to claim your prize.')
    if st.button('Analyze Message') and user_input.strip():
        vec = tfidf.transform([clean_text(user_input)])
        nb_pred = nb_model.predict(vec)[0]
        nb_prob = nb_model.predict_proba(vec)[0]
        tf_res = tf_clf(user_input)[0]
        tf_is_spam = tf_res['label'] in ['LABEL_1', 'spam', 'SPAM']

        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Naive Bayes (TF-IDF)')
            if nb_pred == 1:
                st.error('SPAM / PHISHING')
            else:
                st.success('LEGITIMATE (HAM)')
            st.write(f"Spam Probability: **{nb_prob[1]*100:.2f}%**")
        with col2:
            st.subheader('Transformer (BERT)')
            if tf_is_spam:
                st.error('SPAM / PHISHING')
            else:
                st.success('LEGITIMATE (HAM)')
            st.write(f"Confidence: **{tf_res['score']*100:.2f}%**")

with tab2:
    uploaded_file = st.file_uploader('Upload CSV with a text column', type=['csv'])
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        if 'text' in batch_df.columns:
            cleaned = batch_df['text'].astype(str).apply(clean_text)
            batch_df['Prediction'] = ['Spam' if p == 1 else 'Ham' for p in nb_model.predict(tfidf.transform(cleaned))]
            st.dataframe(batch_df[['text', 'Prediction']])
            st.bar_chart(batch_df['Prediction'].value_counts())
        else:
            st.warning('CSV must contain a column named text.')

with tab3:
    try:
        st.image('evaluation_charts.png')
    except FileNotFoundError:
        st.info('Run train_and_eval.py first.')

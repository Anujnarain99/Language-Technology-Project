import re, joblib, nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from transformers import pipeline

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

df = pd.read_csv('spam.csv').dropna().drop_duplicates()
df['target'] = df['label'].map({'ham': 0, 'spam': 1})

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' url ', text)
    text = re.sub(r'\d+', ' num ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return ' '.join([lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words])

df['clean_text'] = df['text'].apply(preprocess_text)

X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
    df['clean_text'], df['target'], df['text'], test_size=0.2, random_state=42, stratify=df['target']
)

tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_preds = nb_model.predict(X_test_tfidf)

print('--- CLASSICAL BASELINE: TF-IDF + NAIVE BAYES ---')
print(classification_report(y_test, nb_preds, target_names=['Ham', 'Spam']))

joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(nb_model, 'nb_model.pkl')

print('Running Pretrained Transformer on 200 test samples...')
transformer_clf = pipeline('text-classification', model='mrm8488/bert-tiny-finetuned-sms-spam-detection', truncation=True)
sample_texts = raw_test.iloc[:200].tolist()
sample_labels = y_test.iloc[:200].tolist()
tf_results = transformer_clf(sample_texts)
tf_preds = [1 if r['label'] in ['LABEL_1', 'spam', 'SPAM'] else 0 for r in tf_results]

print('\n--- TRANSFORMER MODEL REPORT ---')
print(classification_report(sample_labels, tf_preds, target_names=['Ham', 'Spam']))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(confusion_matrix(y_test, nb_preds), annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=axes[0])
axes[0].set_title('Naive Bayes (TF-IDF) Confusion Matrix')
sns.heatmap(confusion_matrix(sample_labels, tf_preds), annot=True, fmt='d', cmap='Greens', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=axes[1])
axes[1].set_title('Transformer Confusion Matrix (Sample)')
plt.tight_layout()
plt.savefig('evaluation_charts.png')

print('\n--- ERROR ANALYSIS (MISCLASSIFIED SAMPLES) ---')
errors = pd.DataFrame({'Raw_Text': raw_test, 'Actual': y_test, 'Predicted': nb_preds})
for _, row in errors[errors['Actual'] != errors['Predicted']].head(5).iterrows():
    print(f"Actual: {'Spam' if row['Actual']==1 else 'Ham'} | Predicted: {'Spam' if row['Predicted']==1 else 'Ham'} -> {row['Raw_Text']}")

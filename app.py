import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,classification_report
from sklearn.model_selection import train_test_split


# Load the trained model and vectorizer
model = pickle.load(open('sentiment_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
metrics = pickle.load(open('model_metrics.pkl', 'rb'))

data = pd.read_csv('IMDB_Dataset.csv')  
X = data['review']
y = data['sentiment'].map({'positive': 1, 'negative': 0}) 


# Functions for preprocessing
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return clean.sub('', text)

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_punctuation(text):
    PUNCT_TO_REMOVE = string.punctuation
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

def handle_negations(text):
    # List of negation words and contractions
    negations = ["not", "no", "never", "n't", "cannot", "won't", "wouldn't", "couldn't", "shouldn't", "isn't", "wasn't", "doesn't", "don't", "didn't", "can't"]
    words = text.split()
    new_words = []
    skip_next = False
    
    for i in range(len(words) - 1):
        if skip_next:
            skip_next = False
            continue
        if any(neg in words[i] for neg in negations): 
            new_word = words[i] + '_' + words[i + 1] 
            new_words.append(new_word)
            skip_next = True
        else:
            new_words.append(words[i])

    if not skip_next:
        new_words.append(words[-1])

    return ' '.join(new_words)


STOPWORDS = set(stopwords.words('english')) - {"not", "no", "nor", "never"}

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])


st.sidebar.title("Additional Info on the model")
show_metrics = st.sidebar.button("Show Model Performance")
show_info = st.sidebar.button("Show Metrics Explanation")
show_conf_matrix = st.sidebar.button("Show Confusion Matrix")


st.title("🎬 Movie Review Sentiment Analyzer🎬")
st.write("Enter a movie review below to predict whether the sentiment is positive or negative.")

# User input
user_review = st.text_area("Enter your movie review:")

if st.button("Analyze Sentiment"):
    if user_review:
        # Preprocess the input review
        processed_review = user_review.lower()
        processed_review = remove_html_tags(processed_review)
        processed_review = remove_urls(processed_review)
        processed_review = remove_punctuation(processed_review)
        processed_review = remove_stopwords(processed_review)
        processed_review = handle_negations(processed_review)

        # Transform and predict
        review_vector = tfidf.transform([processed_review])
        prediction = model.predict(review_vector)

        # Display the result
        sentiment = "Positive 😊" if prediction[0] == "positive" else "Negative 😞"
        st.success(f"The sentiment of the review is: {sentiment}")
    else:
        st.warning("Please enter a review to analyze.")

# Display model performance if button is clicked
if show_metrics:
    st.subheader("Model Performance Metrics")
    report = metrics['classification_report']
    st.text("              precision    recall  f1-score   support")
    for label in ['negative', 'positive']:
        st.text(f"{label:<12} {report[label]['precision']:<10.2f} {report[label]['recall']:<10.2f} {report[label]['f1-score']:<10.2f} {int(report[label]['support']):<10}")
    st.text("\n    accuracy                           {:.2f}     {}".format(report['accuracy'], int(sum([report[label]['support'] for label in ['negative', 'positive']]))))
    st.text("   macro avg     {:.2f}      {:.2f}      {:.2f}     {}".format(
        report['macro avg']['precision'],
        report['macro avg']['recall'],
        report['macro avg']['f1-score'],
        int(sum([report[label]['support'] for label in ['negative', 'positive']]))
    ))
    st.text("weighted avg    {:.2f}      {:.2f}      {:.2f}     {}".format(
        report['weighted avg']['precision'],
        report['weighted avg']['recall'],
        report['weighted avg']['f1-score'],
        int(sum([report[label]['support'] for label in ['negative', 'positive']]))
    ))

# Display confusion matrix if button is clicked
if show_conf_matrix:
    st.subheader("Confusion Matrix")
    cm = metrics['confusion_matrix']
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(plt.gcf())

if show_info:
    st.subheader("Metrics Explanation")
    st.markdown("- **Precision**: Correct positive predictions out of all positive predictions made.")
    st.markdown("- **Recall**: Correct positive predictions out of all actual positives.")
    st.markdown("- **F1-Score**: Balance between precision and recall.")
    st.markdown("- **Support**: Number of actual occurrences of each class.")
    st.markdown("- **Accuracy**: Overall correctness of the model.")
    st.markdown("- **Macro Avg**: Average of metrics treating all classes equally.")
    st.markdown("- **Weighted Avg**: Average considering class imbalance.")

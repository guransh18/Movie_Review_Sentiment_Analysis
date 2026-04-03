#  Movie Review Sentiment Analyzer

An end-to-end Natural Language Processing (NLP) project that predicts whether a movie review is positive or negative. It features a custom text-cleaning pipeline, a Machine Learning classifier, and an interactive web interface.

##  Features
* **Custom Text Preprocessing:** Handles lowercase conversion, HTML/URL stripping, punctuation removal, and stopword filtering.
* **Contextual Negation Handling:** Custom logic to link negation words to their targets (e.g., "not good" becomes "not_good") to preserve sentiment meaning before vectorization.
* **TF-IDF Vectorization:** Utilizes unigrams, bigrams, and trigrams (`ngram_range=(1, 3)`) to capture complex phrasing.
* **Machine Learning Classifier:** Trained using Logistic Regression for high accuracy on text data.
* **Interactive Web App:** Built with Streamlit, allowing users to type live reviews and instantly see sentiment predictions and word clouds.

## 📁 Project Structure
```text
├── app.py                   # The Streamlit web application
├── Test1.ipynb              # Jupyter notebook with data cleaning, EDA, and model training
├── requirements.txt         # Python dependencies
├── .gitignore               # Files to be ignored by Git
└── README.md                # Project documentation

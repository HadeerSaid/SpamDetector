# Import necessary libraries
import pandas as pd
import numpy as np
import re
import joblib
import streamlit as st
import nltk  
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the dataset 
df = pd.read_csv('spam.csv', encoding='latin-1')
df.drop(columns=['v3', 'v4', 'v5'], inplace=True)


# Data preprocessing
df['v2'] = df['v2'].astype(str)  # Convert to string
df['v2'] = df['v2'].apply(lambda x: re.sub(r'<.*?>', '', x))  # Remove HTML tags
df['v2'] = df['v2'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))  # Remove special characters
df['v2'] = df['v2'].str.lower()  # Convert text to lowercase

# Tokenization and removing stopwords 
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def tokenize_and_remove_stopwords(text):
    if isinstance(text, list):  
        text = ' '.join(text)  
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

df['v2'] = df['v2'].apply(tokenize_and_remove_stopwords)

# Encode labels 
label_encoder = LabelEncoder()
df['v1'] = label_encoder.fit_transform(df['v1'])

# Split the data into training and testing sets
X = df['v2']
y = df['v1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=7000)  # Adjust max_features as needed

# Transform the email text into TF-IDF feature vectors
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Initialize and train the Random Forest model
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train_tfidf, y_train)

# Save the TF-IDF vectorizer and the trained model
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(random_forest_model, 'spam_model.pkl')

# Create a Streamlit app 
st.title("Email Spam Detection ")
email_text = st.text_area("Enter an email:")

#function to classify the email
def classify_email(email_text):
    # Preprocess the input email text
    email_text = re.sub(r'<.*?>', '', email_text)
    email_text = re.sub(r'[^a-zA-Z0-9\s]', '', email_text)
    email_text = email_text.lower()
    email_text = tokenize_and_remove_stopwords(email_text)

    # Transform the email text using the saved TF-IDF vectorizer
    email_text_tfidf = tfidf_vectorizer.transform([email_text])

    # Predict if the email is spam or not
    prediction = random_forest_model.predict(email_text_tfidf)

    return prediction


if st.button("Classify"):
    if email_text:
        prediction = classify_email(email_text)
        if prediction == 1:
            st.write("This is a spam email.")
        else:
            st.write("This is not a spam email.")
    else:
        st.write("Please enter an email.")


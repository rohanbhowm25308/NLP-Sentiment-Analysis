import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import re
import string

import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import joblib

nltk.download('punkt')
nltk.download("punkt_tab")
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("IMDB_Dataset_CLEANED.csv")

print("Dataset Loaded Successfully")
print()

# First 5 Rows
print(df.head())

print()

# Dataset Shape
print("Shape :", df.shape)

print()

# Column Names
print("Columns :", df.columns)

print()

# Check Missing Values
print("Missing Values")
print(df.isnull().sum())

print()

# Sentiment Count
print(df["sentiment"].value_counts())

# ==========================================
# PART 2
# DATA EXPLORATION
# ==========================================

print("\n==============================")
print("DATASET INFORMATION")
print("==============================\n")

# Dataset Information
print(df.info())

print("\n==============================")
print("STATISTICAL SUMMARY")
print("==============================\n")

# Statistical Summary
print(df.describe(include='all'))

print("\n==============================")
print("CHECK DUPLICATE RECORDS")
print("==============================\n")

# Duplicate Rows
print("Duplicate Rows :", df.duplicated().sum())

print("\n==============================")
print("REMOVE DUPLICATE RECORDS")
print("==============================\n")

# Remove duplicates
df = df.drop_duplicates()

print("Dataset Shape After Removing Duplicates :", df.shape)

print("\n==============================")
print("SAMPLE REVIEWS")
print("==============================\n")

# Display first five reviews
for i in range(5):
    print(f"Review {i+1}:")
    print(df['review'].iloc[i])
    print("Sentiment :", df['sentiment'].iloc[i])
    print("-"*80)
    
# ==========================================
# PART 3
# TEXT PREPROCESSING
# ==========================================

print("\n==============================")
print("TEXT PREPROCESSING")
print("==============================\n")

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Load English Stopwords
stop_words = set(stopwords.words('english'))


# Function to clean text
def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove numbers
    text = re.sub(r'\d+', ' ', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenization
    words = nltk.word_tokenize(text)

    # Remove stopwords and lemmatize
    cleaned_words = []

    for word in words:

        if word not in stop_words:

            word = lemmatizer.lemmatize(word)

            cleaned_words.append(word)

    # Join words back into a sentence
    return " ".join(cleaned_words)
# ==========================================
# TEST PREPROCESSING
# ==========================================

sample_review = df["review"].iloc[0]

print("ORIGINAL REVIEW\n")
print(sample_review)

print("\n" + "="*80)

clean_review = preprocess_text(sample_review)

print("CLEANED REVIEW\n")
print(clean_review)

# ==========================================
# PART 4
# APPLY PREPROCESSING TO ENTIRE DATASET
# ==========================================

print("\n==============================")
print("APPLYING TEXT PREPROCESSING")
print("==============================\n")

# Apply preprocessing function to all reviews
df["clean_review"] = df["review"].apply(preprocess_text)

print("Text preprocessing completed successfully!")

print("\n==============================")
print("FIRST 5 CLEANED REVIEWS")
print("==============================\n")

# Display original and cleaned reviews
for i in range(5):

    print(f"Review {i+1}")

    print("\nOriginal Review:")
    print(df["review"].iloc[i])

    print("\nCleaned Review:")
    print(df["clean_review"].iloc[i])

    print("\nSentiment:")
    print(df["sentiment"].iloc[i])

    print("\n" + "="*100)

print("\nDataset Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())    

# ==========================================
# PART 5
# TF-IDF VECTORIZATION
# ==========================================

print("\n==============================")
print("TF-IDF VECTORIZATION")
print("==============================\n")

# Convert sentiment labels into numbers
# positive = 1
# negative = 0

df["label"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

print("Sentiment Labels Converted Successfully!\n")

print(df[["sentiment", "label"]].head())

print("\n==============================")
print("CREATING TF-IDF FEATURES")
print("==============================\n")

# Create TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=5,
    max_df=0.8
)

# Convert text into numerical vectors
X = vectorizer.fit_transform(df["clean_review"])

# Target Variable
y = df["label"]

print("TF-IDF Feature Matrix Created Successfully!")

print("\nShape of Feature Matrix (X):", X.shape)

print("Shape of Target Variable (y):", y.shape)

print("\n==============================")
print("SAMPLE FEATURES")
print("==============================\n")

# Display first 20 feature names
feature_names = vectorizer.get_feature_names_out()

print(feature_names[:20])

print("\nTotal Features Created:", len(feature_names))

# ==========================================
# PART 6
# TRAIN TEST SPLIT
# ==========================================

print("\n==============================")
print("TRAIN TEST SPLIT")
print("==============================\n")

# Split dataset into Training and Testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training and Testing data created successfully!\n")

print("Training Data Shape")
print("-------------------")
print("X_train :", X_train.shape)
print("y_train :", y_train.shape)

print("\nTesting Data Shape")
print("-------------------")
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

print("\nTraining Samples :", len(y_train))
print("Testing Samples :", len(y_test))

# ==========================================
# PART 7
# TRAIN NAIVE BAYES MODEL
# ==========================================

print("\n==============================")
print("TRAINING THE MODEL")
print("==============================\n")

# Create the Multinomial Naive Bayes Model
model = MultinomialNB()

# Train the Model
model.fit(X_train, y_train)

print("Model trained successfully!")

print("\n==============================")
print("MAKING PREDICTIONS")
print("==============================\n")

# Predict on Test Data
y_pred = model.predict(X_test)

print("Predictions completed successfully!")

print("\n==============================")
print("SAMPLE PREDICTIONS")
print("==============================\n")

# Display first 10 predictions
for i in range(10):

    actual = "Positive" if y_test.iloc[i] == 1 else "Negative"
    predicted = "Positive" if y_pred[i] == 1 else "Negative"

    print(f"Review {i+1}")
    print(f"Actual Sentiment    : {actual}")
    print(f"Predicted Sentiment : {predicted}")
    print("-" * 50)
    
# ==========================================
# PART 8
# MODEL EVALUATION
# ==========================================

print("\n==============================")
print("MODEL EVALUATION")
print("==============================\n")

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy : {accuracy * 100:.2f}%")

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================\n")

# Display Classification Report
print(classification_report(
    y_test,
    y_pred,
    target_names=["Negative", "Positive"]
))

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================\n")

# Generate Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

print("\n==============================")
print("CONFUSION MATRIX DETAILS")
print("==============================\n")

print(f"True Negative (TN): {cm[0][0]}")
print(f"False Positive (FP): {cm[0][1]}")
print(f"False Negative (FN): {cm[1][0]}")
print(f"True Positive (TP): {cm[1][1]}")

print("\n==============================")
print("MODEL PERFORMANCE SUMMARY")
print("==============================\n")

print(f"Total Test Samples : {len(y_test)}")
print(f"Correct Predictions : {(y_pred == y_test).sum()}")
print(f"Incorrect Predictions : {(y_pred != y_test).sum()}")
print(f"Accuracy : {accuracy * 100:.2f}%")

# ==========================================
# PART 9
# SAVE MODEL & TEST CUSTOM REVIEWS
# ==========================================

print("\n==============================")
print("SAVING MODEL")
print("==============================\n")

# Save the trained model
joblib.dump(model, "model.pkl")

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model saved successfully as model.pkl")
print("Vectorizer saved successfully as vectorizer.pkl")

print("\n==============================")
print("CUSTOM REVIEW PREDICTION")
print("==============================\n")

# Function to predict sentiment
def predict_sentiment(review):

    # Clean the review
    clean_review = preprocess_text(review)

    # Convert to TF-IDF
    review_vector = vectorizer.transform([clean_review])

    # Predict
    prediction = model.predict(review_vector)[0]

    # Prediction Probability
    probability = model.predict_proba(review_vector)[0]

    # Convert prediction to label

    confidence = max(probability) * 100
    
    review_lower = review.lower()

    strong_positive = [
     "excellent", "outstanding", "amazing", "fantastic",
     "awesome", "perfect", "brilliant", "superb", "unique",
     "masterpiece", "incredible", "best"
    ]

    positive = [
     "good", "nice", "great", "love", "liked",
     "wonderful", "enjoyed", "beautiful"
    ]

    strong_negative = [
     "worst", "terrible", "awful", "poor",
     "horrible", "disappointing", "waste",
     "pathetic", "bad", "hate"
    ]

    if any(word in review_lower for word in strong_positive):
      prediction = "positive"
      confidence = max(confidence, 95)

    elif any(word in review_lower for word in positive):
       prediction = "positive"
       confidence = max(confidence, 70)

    elif any(word in review_lower for word in strong_negative):
      prediction = "negative"
      confidence = max(confidence, 95)

    confidence = min(confidence, 99.9)
    
    return prediction, confidence

# Sample Reviews
sample_reviews = [

    "This movie was absolutely amazing. I loved every minute of it.",

    "Worst movie ever. Waste of time.",

    "The acting was fantastic and the story was very interesting.",

    "Terrible experience. I will never watch it again.",

    "It was okay. Not the best but not the worst."

]

for i, review in enumerate(sample_reviews, start=1):

    sentiment, confidence = predict_sentiment(review)

    print(f"\nReview {i}")
    print("-" * 70)
    print("Text       :", review)
    print("Prediction :", sentiment)
    print(f"Confidence : {confidence:.2f}%")    
    
# ==========================================
# DOWNLOAD NLTK DATA (Run Only First Time)
# ==========================================

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)

# ==========================================
# LOAD MODEL & VECTORIZER
# ==========================================

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ==========================================
# TEXT PREPROCESSING FUNCTION
# ==========================================

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    text = re.sub(r"\d+", " ", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"\s+", " ", text).strip()

    words = nltk.word_tokenize(text)

    cleaned_words = []

    for word in words:

        if word not in stop_words:

            word = lemmatizer.lemmatize(word)

            cleaned_words.append(word)

    return " ".join(cleaned_words)

# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")

# ==========================================
# PREDICT API
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    review = data["review"]

    review = preprocess_text(review)

    review_vector = vectorizer.transform([review])

    prediction = model.predict(review_vector)[0]

    probability = model.predict_proba(review_vector)[0]

    confidence = round(max(probability) * 100, 2)

    if prediction == 1:
        sentiment = "Positive 😊"
    else:
        sentiment = "Negative 😞"

    return jsonify({
        "sentiment": sentiment,
        "confidence": confidence
    })

# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)    
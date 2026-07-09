# ==========================================
# IMPORT LIBRARIES
# ==========================================

import pandas as pd
import re
import string
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================
# DOWNLOAD NLTK DATA
# ==========================================

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4")

# ==========================================
# LOAD DATASET
# ==========================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv("IMDB_Dataset_CLEANED.csv")

print("Dataset Loaded Successfully")
print("Shape :", df.shape)

# ==========================================
# REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates()

print("Dataset Shape After Removing Duplicates :", df.shape)

# ==========================================
# NLP TOOLS
# ==========================================

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))

# ==========================================
# TEXT PREPROCESSING FUNCTION
# ==========================================

def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    text = re.sub(r"\d+", " ", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\s+", " ", text).strip()

    words = nltk.word_tokenize(text)

    cleaned_words = []

    for word in words:

        if word not in stop_words:

            word = lemmatizer.lemmatize(word)

            cleaned_words.append(word)

    return " ".join(cleaned_words)

# ==========================================
# APPLY PREPROCESSING
# ==========================================

print("\nApplying Text Preprocessing...")

df["clean_review"] = df["review"].apply(preprocess_text)

print("Text Preprocessing Completed Successfully!")

print("\nSample Cleaned Review:\n")

print(df["clean_review"].iloc[0])

# ==========================================
# CONVERT LABELS
# ==========================================

print("\n" + "=" * 60)
print("CONVERTING SENTIMENT LABELS")
print("=" * 60)

df["label"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

print("Labels Converted Successfully!")

print(df[["sentiment", "label"]].head())

# ==========================================
# TF-IDF VECTORIZATION
# ==========================================

print("\n" + "=" * 60)
print("CREATING TF-IDF FEATURES")
print("=" * 60)

vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=5,
    max_df=0.8
)

X = vectorizer.fit_transform(df["clean_review"])

y = df["label"]

print("TF-IDF Features Created Successfully!")

print("Feature Matrix Shape :", X.shape)

print("Target Shape :", y.shape)

print("\nTotal Features :", len(vectorizer.get_feature_names_out()))

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

print("\n" + "=" * 60)
print("TRAIN TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Samples :", len(y_train))
print("Testing Samples  :", len(y_test))

print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
# ==========================================
# TRAIN MULTINOMIAL NAIVE BAYES MODEL
# ==========================================

from sklearn.naive_bayes import MultinomialNB

print("\n" + "=" * 60)
print("TRAINING MACHINE LEARNING MODEL")
print("=" * 60)

# Create Model
model = MultinomialNB()

# Train Model
model.fit(X_train, y_train)

print("Model Trained Successfully!")

# ==========================================
# MAKE PREDICTIONS
# ==========================================

print("\n" + "=" * 60)
print("MAKING PREDICTIONS")
print("=" * 60)

# Predict Test Data
y_pred = model.predict(X_test)

print("Prediction Completed Successfully!")

# ==========================================
# SAMPLE PREDICTIONS
# ==========================================

print("\nSample Predictions\n")

for i in range(10):

    actual = "Positive" if y_test.iloc[i] == 1 else "Negative"

    predicted = "Positive" if y_pred[i] == 1 else "Negative"

    print(f"Review {i+1}")

    print(f"Actual Sentiment    : {actual}")

    print(f"Predicted Sentiment : {predicted}")

    print("-" * 60)

# ==========================================
# MODEL INFORMATION
# ==========================================

print("\n" + "=" * 60)
print("MODEL INFORMATION")
print("=" * 60)

print("Algorithm :", model.__class__.__name__)

print("Number of Training Samples :", len(y_train))

print("Number of Testing Samples :", len(y_test))

print("Training Completed Successfully!")
# ==========================================
# MODEL EVALUATION
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy : {accuracy * 100:.2f}%")

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(classification_report(
    y_test,
    y_pred,
    target_names=["Negative", "Positive"]
))

# ==========================================
# CONFUSION MATRIX
# ==========================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(y_test, y_pred)

print(cm)

# ==========================================
# CONFUSION MATRIX DETAILS
# ==========================================

print("\nTrue Negative (TN):", cm[0][0])
print("False Positive (FP):", cm[0][1])
print("False Negative (FN):", cm[1][0])
print("True Positive (TP):", cm[1][1])

# ==========================================
# PERFORMANCE SUMMARY
# ==========================================

correct_predictions = (y_pred == y_test).sum()
incorrect_predictions = (y_pred != y_test).sum()

print("\n" + "=" * 60)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 60)

print(f"Total Test Samples      : {len(y_test)}")
print(f"Correct Predictions     : {correct_predictions}")
print(f"Incorrect Predictions   : {incorrect_predictions}")
print(f"Accuracy                : {accuracy * 100:.2f}%")

# ==========================================
# DISPLAY SAMPLE REVIEWS
# ==========================================

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

for i in range(5):

    review = df.iloc[y_test.index[i]]["review"][:120]

    actual = "Positive" if y_test.iloc[i] == 1 else "Negative"

    predicted = "Positive" if y_pred[i] == 1 else "Negative"

    print(f"\nReview {i+1}")
    print("-" * 60)
    print("Review    :", review + "...")
    print("Actual    :", actual)
    print("Predicted :", predicted)

print("\nModel Evaluation Completed Successfully!")
# ==========================================
# SAVE MODEL & VECTORIZER
# ==========================================

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

joblib.dump(model, "model.pkl")

joblib.dump(vectorizer, "vectorizer.pkl")

print("✓ model.pkl saved successfully!")

print("✓ vectorizer.pkl saved successfully!")

# ==========================================
# CUSTOM REVIEW PREDICTION FUNCTION
# ==========================================

def predict_sentiment(review):

    clean_review = preprocess_text(review)

    review_vector = vectorizer.transform([clean_review])

    prediction = model.predict(review_vector)[0]

    probability = model.predict_proba(review_vector)[0]

    confidence = round(max(probability) * 100, 2)

    sentiment = "Positive" if prediction == 1 else "Negative"

    return sentiment, confidence

# ==========================================
# TEST CUSTOM REVIEWS
# ==========================================

print("\n" + "=" * 60)
print("TESTING CUSTOM REVIEWS")
print("=" * 60)

sample_reviews = [

    "This movie was absolutely amazing. I loved every minute of it.",

    "Worst movie ever. Waste of time.",

    "The acting was fantastic and the story was very interesting.",

    "Terrible experience. I will never watch it again.",

    "It was okay. Not the best but not the worst.",

    "Excellent movie with outstanding acting and direction.",

    "Poor storyline and horrible acting."

]

for i, review in enumerate(sample_reviews, start=1):

    sentiment, confidence = predict_sentiment(review)

    print(f"\nReview {i}")

    print("-" * 70)

    print("Text       :", review)

    print("Prediction :", sentiment)

    print(f"Confidence : {confidence:.2f}%")

# ==========================================
# TRAINING COMPLETED
# ==========================================

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated Files:")

print("✓ model.pkl")

print("✓ vectorizer.pkl")

print("\nThese files are now ready to be used by app.py")

print("\nYou can now start your Flask application using:")

print("\npython app.py")
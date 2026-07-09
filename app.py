# ==========================================
# IMPORT LIBRARIES
# ==========================================

from flask import Flask, render_template, request, jsonify

import joblib
import nltk
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==========================================
# DOWNLOAD NLTK DATA
# (Downloads only if missing)
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
# CREATE FLASK APP
# ==========================================

app = Flask(__name__)

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

print("Model Loaded Successfully")
print("Vectorizer Loaded Successfully")

# ==========================================
# NLP TOOLS
# ==========================================

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

    review = data.get("review", "").strip()

    # Empty Review Check
    if review == "":

        return jsonify({
            "sentiment": "Please enter a review.",
            "confidence": 0
        })

    # --------------------------------------
    # PREPROCESS TEXT
    # --------------------------------------

    clean_review = preprocess_text(review)

    review_vector = vectorizer.transform([clean_review])

    # --------------------------------------
    # MODEL PREDICTION
    # --------------------------------------

    prediction = model.predict(review_vector)[0]

    probability = model.predict_proba(review_vector)[0]

    confidence = round(max(probability) * 100, 2)

    review_lower = review.lower()

    # --------------------------------------
    # CUSTOM CONFIDENCE RULES
    # --------------------------------------

    strong_positive = [

        "excellent",
        "outstanding",
        "amazing",
        "fantastic",
        "awesome",
        "perfect",
        "brilliant",
        "superb",
        "unique",
        "masterpiece",
        "incredible",
        "best"

    ]

    positive = [

        "good",
        "great",
        "nice",
        "love",
        "liked",
        "wonderful",
        "beautiful",
        "enjoyed"

    ]

    strong_negative = [

        "worst",
        "terrible",
        "awful",
        "poor",
        "horrible",
        "waste",
        "pathetic",
        "bad",
        "hate",
        "disappointing"

    ]

    # --------------------------------------
    # APPLY CUSTOM RULES
    # --------------------------------------

    if any(word in review_lower for word in strong_positive):

        prediction = 1
        confidence = max(confidence, 95)

    elif any(word in review_lower for word in positive):

        prediction = 1
        confidence = max(confidence, 70)

    elif any(word in review_lower for word in strong_negative):

        prediction = 0
        confidence = max(confidence, 95)

    confidence = min(confidence, 99.9)

    # --------------------------------------
    # FINAL SENTIMENT
    # --------------------------------------

    if prediction == 1:

        sentiment = "Positive 😊"

    else:

        sentiment = "Negative 😞"

    # --------------------------------------
    # RETURN RESULT
    # --------------------------------------

    return jsonify({

        "sentiment": sentiment,

        "confidence": round(confidence, 2)

    })
    
# ==========================================
# RUN FLASK APP
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )    
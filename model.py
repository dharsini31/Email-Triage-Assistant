import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import re

MODEL_PATH = "models/email_classifier.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"


# -------------------------------
# TEXT PREPROCESSING
# -------------------------------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# -------------------------------
# SAMPLE DATASET
# -------------------------------
def create_sample_dataset():
    emails = [
        ("Meeting scheduled tomorrow at 2 PM", "Work"),
        ("Project deadline approaching", "Work"),
        ("Please review the document", "Work"),
        ("Let's catch up this weekend", "Personal"),
        ("Happy birthday! Hope you enjoy", "Personal"),
        ("Dinner plan tonight?", "Personal"),
        ("URGENT: Server is down", "Urgent"),
        ("Critical bug fix ASAP", "Urgent"),
        ("Emergency meeting now", "Urgent"),
        ("You won $1,000,000!", "Spam"),
        ("Limited time offer buy now", "Spam"),
        ("Free gift card click here", "Spam"),
    ]
    emails = [
        # Work emails
        ("Meeting scheduled for tomorrow at 2 PM. Please confirm your attendance.", "Work"),
        ("Project deadline is approaching. Need status update by end of week.", "Work"),
        ("Please review the attached document and provide feedback.", "Work"),
        ("Team meeting notes from today's session.", "Work"),
        ("Budget approval required for Q4 marketing campaign.", "Work"),
        ("Client presentation scheduled for next Monday.", "Work"),
        ("Code review needed for the new feature implementation.", "Work"),
        ("Quarterly report submission due next Friday.", "Work"),
        
        # Personal emails
        ("Hey! How are you doing? Let's catch up soon.", "Personal"),
        ("Thanks for the birthday wishes! Really appreciate it.", "Personal"),
        ("Are we still on for dinner this weekend?", "Personal"),
        ("Just wanted to say hello and see how you're doing.", "Personal"),
        ("Family gathering next month. Hope you can make it!", "Personal"),
        ("Thanks for helping me move last weekend!", "Personal"),
        ("Let's plan a trip together. What do you think?", "Personal"),
        ("Happy holidays! Wishing you all the best.", "Personal"),
        
        # Urgent emails
        ("URGENT: Server is down. Need immediate attention!", "Urgent"),
        ("Critical bug found in production. Fix ASAP!", "Urgent"),
        ("Deadline moved up to tomorrow. Please prioritize.", "Urgent"),
        ("Emergency meeting in 30 minutes. Your presence required.", "Urgent"),
        ("URGENT: Client complaint needs immediate resolution.", "Urgent"),
        ("Critical security issue detected. Action required now.", "Urgent"),
        ("Deadline is today! Please submit your work immediately.", "Urgent"),
        ("ASAP: System failure affecting all users.", "Urgent"),
        
        # Spam emails
        ("Congratulations! You've won $1,000,000! Click here to claim.", "Spam"),
        ("Limited time offer! Buy now and get 90% discount!", "Spam"),
        ("You have been selected for a special prize. Act now!", "Spam"),
        ("Make money fast! No experience needed. Start today!", "Spam"),
        ("Free gift card! Click this link to redeem instantly.", "Spam"),
        ("You've won an iPhone! Claim your prize now!", "Spam"),
        ("Exclusive deal just for you! Don't miss out!", "Spam"),
        ("Act now! Limited time offer expires soon!", "Spam"),
    ]
   return pd.DataFrame(emails, columns=["email", "category"])


# -------------------------------
# TRAIN MODEL
# -------------------------------
def train_email_classifier():
    df = create_sample_dataset()
    df["processed"] = df["email"].apply(preprocess_text)

    X = df["processed"]
    y = df["category"]

    vectorizer = TfidfVectorizer(stop_words="english")
    X_vectors = vectorizer.fit_transform(X)

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_vectors, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(classifier, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("Model trained and saved.")


# -------------------------------
# LOAD MODEL
# -------------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        train_email_classifier()

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


# -------------------------------
# PREDICT CATEGORY
# -------------------------------
def predict_category(email_text):
    model, vectorizer = load_model()

    processed = preprocess_text(email_text)
    vector = vectorizer.transform([processed])

    prediction = model.predict(vector)[0]
    confidence = np.max(model.predict_proba(vector))

    return prediction, round(float(confidence), 2)


# -------------------------------
# SUMMARIZATION (Simple)
# -------------------------------
def summarize_email(email_text):
    sentences = email_text.split(".")
    summary = sentences[0]
    if len(sentences) > 1:
        summary += "."
    return summary.strip()


# -------------------------------
# URGENCY DETECTION
# -------------------------------
def detect_urgency(email_text):
    urgent_keywords = ["urgent", "asap", "immediately", "critical", "emergency"]

    email_lower = email_text.lower()
    for word in urgent_keywords:
        if word in email_lower:
            return "High"

    return "Normal"


def generate_reply(category):
    replies = {
        "Work": "Thank you for the update. I will review and respond shortly.",
        "Personal": "Thanks for reaching out! Looking forward to it.",
        "Urgent": "I acknowledge the urgency. I will address this immediately.",
        "Spam": "This message appears suspicious and will not be responded to."
    }

    return replies.get(category, "Thank you for your email.")
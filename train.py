import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def train_email_classifier():
    emails = [
        ("Meeting tomorrow at 2 PM", "Work"),
        ("Project deadline approaching", "Work"),
        ("Client presentation next week", "Work"),
        ("Let's catch up soon", "Personal"),
        ("Happy birthday!", "Personal"),
        ("Dinner this weekend?", "Personal"),
        ("URGENT: Server down", "Urgent"),
        ("Critical issue needs fix ASAP", "Urgent"),
        ("Emergency meeting now", "Urgent"),
        ("You won $1,000,000!", "Spam"),
        ("Limited time offer!", "Spam"),
        ("Click here to claim prize", "Spam"),
    ]

    df = pd.DataFrame(emails, columns=["email", "category"])

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df["email"])
    y = df["category"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/email_classifier.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

    return True

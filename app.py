from flask import Flask, render_template, request, jsonify
import pickle
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np

app = Flask(__name__)

# Load models
MODEL_PATH = 'models/email_classifier.pkl'
VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'

# Initialize models (will be loaded if they exist)
classifier = None
vectorizer = None

def load_models():
    global classifier, vectorizer
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        classifier = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return True
    return False

# Load models on startup
load_models()

def preprocess_text(text):
    """Clean and preprocess email text"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_urgency(text):
    """Detect urgency level based on keywords"""
    urgency_keywords = ['urgent', 'asap', 'deadline', 'immediate', 'critical', 
                       'important', 'emergency', 'rush', 'time-sensitive', 'hurry']
    text_lower = text.lower()
    urgency_score = sum(1 for keyword in urgency_keywords if keyword in text_lower)
    
    if urgency_score >= 3:
        return "High", urgency_score
    elif urgency_score >= 1:
        return "Medium", urgency_score
    else:
        return "Low", urgency_score

def summarize_email(text, max_sentences=3):
    """Simple extractive summarization"""
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    if len(sentences) <= max_sentences:
        return '. '.join(sentences) + '.'
    
    # Take first and last sentences
    summary = sentences[:max_sentences//2] + sentences[-max_sentences//2:]
    return '. '.join(summary) + '.'

def generate_smart_reply(category, email_text):
    """Generate smart reply suggestions based on category"""
    replies = {
        'Work': [
            "Thank you for your email. I'll review this and get back to you shortly.",
            "I've received your message and will address this matter promptly.",
            "Thanks for reaching out. Let me check on this and revert."
        ],
        'Personal': [
            "Thanks for your message! I'll respond soon.",
            "Great to hear from you! I'll get back to you shortly.",
            "Thanks for reaching out. Talk soon!"
        ],
        'Urgent': [
            "I understand this is urgent. I'm looking into it right away.",
            "Received your urgent message. I'll prioritize this immediately.",
            "Noted the urgency. I'll handle this as a top priority."
        ],
        'Spam': [
            "This email appears to be spam and will be filtered.",
            "Spam detected. No action required.",
            "This message has been flagged as spam."
        ]
    }
    
    # Select appropriate reply based on category
    category_replies = replies.get(category, replies['Work'])
    
    # Add context-aware reply if urgency detected
    urgency_level, _ = detect_urgency(email_text)
    if urgency_level == "High" and category != "Spam":
        return category_replies[0] + " I'll prioritize this immediately."
    
    return category_replies[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_email():
    try:
        data = request.json
        email_text = data.get('email', '')
        
        if not email_text:
            return jsonify({'error': 'No email text provided'}), 400
        
        # Preprocess text
        processed_text = preprocess_text(email_text)
        
        # Check if model is loaded
        if classifier is None or vectorizer is None:
            return jsonify({
                'error': 'Model not trained yet. Please train the model first.',
                'category': 'Unknown',
                'summary': summarize_email(email_text),
                'urgency': detect_urgency(email_text)[0],
                'smart_reply': 'Please train the model first.'
            }), 200
        
        # Vectorize text
        text_vector = vectorizer.transform([processed_text])
        
        # Predict category
        category = classifier.predict(text_vector)[0]
        confidence = max(classifier.predict_proba(text_vector)[0])
        
        # Detect urgency
        urgency_level, urgency_score = detect_urgency(email_text)
        
        # Generate summary
        summary = summarize_email(email_text)
        
        # Generate smart reply
        smart_reply = generate_smart_reply(category, email_text)
        
        return jsonify({
            'category': category,
            'confidence': round(confidence * 100, 2),
            'summary': summary,
            'urgency': urgency_level,
            'urgency_score': urgency_score,
            'smart_reply': smart_reply
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


    try:
        # This would typically load from a dataset
        # For now, we'll use a simple training approach
        from train_model import train_email_classifier
        
        result = train_email_classifier()
        
        if result:
            # Reload models
            load_models()
            return jsonify({'message': 'Model trained successfully!'}), 200
        else:
            return jsonify({'error': 'Training failed'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    app.run(debug=True, port=5000)
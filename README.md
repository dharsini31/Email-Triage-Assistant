#  Email Triage Assistant

An AI-powered Email Triage Assistant that intelligently categorizes, summarizes, and generates smart replies for emails using Machine Learning and NLP.

---

 Project Overview

Managing large volumes of emails manually is inefficient. This project builds an AI Agent that:

- Categorizes emails (Work, Personal, Spam, Urgent)
- Summarizes long email threads
- Suggests automated smart replies
- Detects urgency level

---

##  Technologies Used

- Python
- Flask
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- HTML/CSS

---

How It Works

1. User pastes email content
2. Text is converted to TF-IDF vectors
3. Logistic Regression model predicts category
4. Email is summarized
5. Auto-reply suggestion is generated

---

 Model Details

- Algorithm: Logistic Regression
- Text Processing: TF-IDF
- Classification Categories:
  - Work
  - Personal
  - Spam
  - Urgent

---

 Unique Features

- Automatic Email Summarization
- Smart Reply Suggestion System
- Urgency Detection Logic
- AI Agent-based workflow

---

## 🛠 Installation

```bash
git clone https://github.com/dharsini31/Email-Triage-Assistant
cd Email-Triage-Assistant
pip install -r requirements.txt
python app.py

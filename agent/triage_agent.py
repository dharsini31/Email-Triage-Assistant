from agent.classifier import EmailClassifier
from agent.summarizer import EmailSummarizer
from agent.urgency import EmailUrgency
from agent.responder import EmailResponder


class EmailTriageAgent:
    def __init__(self):
        self.classifier = EmailClassifier()
        self.summarizer = EmailSummarizer()
        self.urgency = EmailUrgency()
        self.responder = EmailResponder()

    def process_email(self, text):
        category = self.classifier.classify(text)
        summary = self.summarizer.summarize(text)
        urgency = self.urgency.detect(text)
        response = self.responder.generate(text)

        return {
            "category": category,
            "summary": summary,
            "urgency": urgency,
            "response": response
        }

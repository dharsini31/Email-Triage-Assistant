from transformers import pipeline

class EmailClassifier:
    def __init__(self):
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )

    def classify(self, text):
        labels = ["Work", "Personal", "Spam", "Support"]
        result = self.classifier(text, labels)
        return result["labels"][0]

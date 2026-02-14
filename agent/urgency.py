class EmailUrgency:
    def __init__(self):
        self.high_keywords = [
            "urgent",
            "asap",
            "immediately",
            "serious",
            "warning",
            "deadline",
            "final",
            "important",
            "priority",
            "delaying",
            "qualifying",
            "last chance",
            "critical",
            "action required"
        ]

        self.medium_keywords = [
            "review",
            "update",
            "notice",
            "reminder",
            "attention",
            "fix",
            "improve",
            "issue"
        ]

    def detect(self, text):
        text = text.lower()
        score = 0

        # High urgency words add more score
        for word in self.high_keywords:
            if word in text:
                score += 2

        # Medium urgency words add smaller score
        for word in self.medium_keywords:
            if word in text:
                score += 1

        # Decision logic
        if score >= 4:
            return "High"
        elif score >= 2:
            return "Medium"
        else:
            return "Low"

class EmailSummarizer:
    def summarize(self, text):
        sentences = text.split(".")
        if len(sentences) > 2:
            return ".".join(sentences[:2]) + "."
        return text

from collections import deque
import tiktoken

class HistoryManager:

    def __init__(self, max_entries=5):
        self.history = deque(maxlen=max_entries)

    def compress_text(self, text, max_tokens=150):
        encoding = tiktoken.encoding_for_model("gpt-4")
        tokens = encoding.encode(text)

        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]

        return encoding.decode(tokens)

    def add_entry(self, query: str, result: dict):
        compressed_result = self.compress_text(str(result))

        self.history.append({
            "query": query,
            "compressed_result": compressed_result
        })

    def get_context(self):
        return list(self.history)

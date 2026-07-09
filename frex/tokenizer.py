class BasicTokenizer:
    def __init__(self):
        # A simple character-level vocabulary
        self.vocab = {}
        self.inverse_vocab = {}
        self.vocab_size = 0

    def fit(self, text: str):
        """Learns the vocabulary from a dataset."""
        unique_chars = sorted(list(set(text)))
        for idx, char in enumerate(unique_chars):
            self.vocab[char] = idx
            self.inverse_vocab[idx] = char
        self.vocab_size = len(self.vocab)

    def encode(self, text: str) -> list:
        """Converts text into a list of integer IDs."""
        return [self.vocab[char] for char in text if char in self.vocab]

    def decode(self, ids: list) -> str:
        """Converts integer IDs back into human-readable text."""
        return "".join([self.inverse_vocab[idx] for idx in ids if idx in self.inverse_vocab])
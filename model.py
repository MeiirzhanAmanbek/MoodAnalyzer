# model.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class VibeModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression(max_iter=1000)

        self.texts = [
            "I love this so much",
            "This is amazing",
            "I'm very happy today",
            "This is terrible",
            "I hate everything",
            "I'm so angry right now",
            "It's okay, nothing special",
            "I feel neutral about this"
        ]

        self.labels = [
            "happy",
            "happy",
            "happy",
            "angry",
            "angry",
            "angry",
            "neutral",
            "neutral"
        ]

        self._train()

    def _train(self):
        X = self.vectorizer.fit_transform(self.texts)
        self.model.fit(X, self.labels)

    def predict(self, text):
        vec = self.vectorizer.transform([text])
        pred = self.model.predict(vec)[0]
        prob = max(self.model.predict_proba(vec)[0])
        return pred, prob

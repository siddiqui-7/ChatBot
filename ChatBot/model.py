import json
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

class IntentClassifier:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.model = None
        self.vectorizer = None
        self.load_data()

    def load_data(self):
        with open(self.dataset_path, 'r') as file:
            self.data = json.load(file)
        self.intents = self.data['intents']
        self.train_model()

    def train_model(self):
        X = []
        y = []
        for intent in self.intents:
            for pattern in intent['patterns']:
                X.append(pattern)
                y.append(intent['tag'])
        
        self.vectorizer = TfidfVectorizer()
        X_tfidf = self.vectorizer.fit_transform(X)
        
        self.model = LogisticRegression()
        self.model.fit(X_tfidf, y)

    def predict_intent(self, user_input):
        X_input = self.vectorizer.transform([user_input])
        predicted_intent = self.model.predict(X_input)[0]
        return predicted_intent

    def get_response(self, intent):
        for intent_data in self.intents:
            if intent_data['tag'] == intent:
                return random.choice(intent_data['responses'])
        return "I'm not sure I understand. Could you clarify?"


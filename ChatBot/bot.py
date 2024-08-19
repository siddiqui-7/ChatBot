from model import IntentClassifier

class CustomerServiceBot:
    def __init__(self, model):
        self.model = model

    def get_response(self, user_input):
        intent = self.model.predict_intent(user_input)
        response = self.model.get_response(intent)
        return response

import streamlit as st
from bot import CustomerServiceBot
from model import IntentClassifier

def main():
    st.title("CODEXCUE Customer Service Bot")

    intent_classifier = IntentClassifier(dataset_path='intents.json')
    bot = CustomerServiceBot(model=intent_classifier)

    user_input = st.text_input("You: ")
    if user_input:
        response = bot.get_response(user_input)
        st.write(f"Bot: {response}")

if __name__ == "__main__":
    main()

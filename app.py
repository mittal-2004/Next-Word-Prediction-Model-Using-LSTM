import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load files
model = load_model("lstm_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

# Function for prediction
def predict_next_word(text):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')

    predicted = np.argmax(model.predict(token_list), axis=-1)

    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word
    return ""

# Streamlit UI
st.title("🧠 Next Word Prediction Model")
st.write("Type a sentence and the model will predict the next word.")

input_text = st.text_input("Enter your text:")

if st.button("Predict"):
    if input_text:
        next_word = predict_next_word(input_text)
        st.success(f"Next word: **{next_word}**")
    else:
        st.warning("Please enter some text")
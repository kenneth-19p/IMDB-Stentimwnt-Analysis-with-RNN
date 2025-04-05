import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.models import load_model

# Step 1: Load the IMDB dataset and word index

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

## Step 2: Load the pre-trained model with RELU activation function
model = load_model('imdb_rnn_model.h5')

## Step 3 : Helper Functions to decode the reviews and convert them to sequences
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

## Function to preprocess user input
def preprocess_input(input_text):
    # Tokenize the input text
    tokens = input_text.lower().split()
    # Convert tokens to their corresponding word indices
    encoded_review = [word_index.get(word, 2) + 3 for word in tokens]
    # Pad the sequence to ensure uniform length
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


## Streanlit Web App
import streamlit as st

st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (Positive/Negative):") 

# User input
user_review = st.text_area("Movie Review")

if st.button('Classify'):

    preprocess_input=preprocess_input(user_review)

    # Predict sentiment
    predict = model.predict(preprocess_input)
    sentiment = 'Positive' if predict[0][0] > 0.5 else 'Negative'

    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {predict[0][0]}")

else:
    st.write('Please enter a movie review.')
import streamlit as st
import pickle
import numpy as np

# Load the saved model and vectorizer
with open('spam_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    cv = pickle.load(vectorizer_file)

# Streamlit App
st.title("Spam Comment Detection")

# Text input for comment
user_input = st.text_area("Enter a comment to check if it's spam:")

# Predict button
if st.button("Check Spam"):
    if user_input:
        # Vectorize the user input
        input_vector = cv.transform([user_input]).toarray()

        # Make the prediction
        prediction = model.predict(input_vector)

        # Display the result
        if prediction == 1:
            st.error("This is likely a spam comment.")
        else:
            st.success("This seems like a legitimate comment.")
    else:
        st.warning("Please enter a comment to check.")

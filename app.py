import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load("svm_text_detection_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page title
st.title("AI Generated Text Detection")
st.write("Enter text below to check whether it is AI-generated or Human-written.")

# Text input
user_input = st.text_area("Enter your text here:", height=200)

# Predict button
if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize input
        input_vec = vectorizer.transform([user_input])
        
        # Prediction
        prediction = model.predict(input_vec)[0]
        
        # Output
        if prediction == 1:
            st.success("🧠 This text is AI-generated")
        else:
            st.success("👤 This text is Human-written")

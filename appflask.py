import os
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
model_path = os.path.join(BASE_DIR, "model", "svm_text_detection_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl")

# Load model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""

    if request.method == "POST":
        text = request.form["text"]
        text_vec = vectorizer.transform([text])
        result = model.predict(text_vec)

        prediction = "AI Generated Text" if result[0] == 1 else "Human Written Text"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

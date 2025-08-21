from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load ML Model
model = pickle.load(open("models/career_model.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        user_input = request.form.to_dict()
        data = np.array([float(v) for v in user_input.values()]).reshape(1, -1)
        
        # Predict Career
        prediction = model.predict(data)[0]

        # Career Dictionary
        jobs = {
            0: "AI/ML Specialist",
            1: "Data Scientist",
            2: "Software Developer",
            3: "Cyber Security Analyst",
            4: "Project Manager",
            5: "Business Analyst",
            6: "Networking Engineer",
            7: "Database Administrator"
        }

        return render_template("results.html", career=jobs[prediction])

if __name__ == "__main__":
    app.run(debug=True)

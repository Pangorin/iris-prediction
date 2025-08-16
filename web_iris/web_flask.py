from flask import Flask, render_template, request
import pickle
import numpy as np

with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)
    
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST":
        # Lấy input từ form
        sepal_length = float(request.form["sepal_length"])
        sepal_width = float(request.form["sepal_width"])
        petal_length = float(request.form["petal_length"])
        petal_width = float(request.form["petal_width"])

        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        pred = model.predict(features)[0]
        classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
        prediction = classes[pred]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
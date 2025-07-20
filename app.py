from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([features])
        iris_classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        result = iris_classes[prediction[0]]
        return render_template('index.html', prediction_text=f"Predicted Iris Class: {result}")
    except:
        return render_template('index.html', prediction_text="Invalid input!")

# ✅ THIS PART IS MANDATORY
if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)

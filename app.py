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
if _name_ == '_main_':
    print("Starting Flask app...")
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

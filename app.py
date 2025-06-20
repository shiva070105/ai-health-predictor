from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model/heart_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_input = np.array(features).reshape(1, -1)
    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0][1] * 100

    return render_template('result.html', 
        prediction="High Risk" if prediction == 1 else "Low Risk",
        probability=round(probability, 2)
    )

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
with open('house_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [
            float(request.form['grade']),
            float(request.form['living_area']),
            float(request.form['basement_area']),
            float(request.form['bathrooms']),
            float(request.form['year_built']),
        ]
        prediction = model.predict([data])[0]
        return render_template('index.html', prediction_text=f'Estimated Price: ₹{prediction:,.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text='Invalid input!')

if __name__ == '__main__':
    app.run(debug=True)

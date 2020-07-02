# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 10:59:24 2020

@author: patak
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    result = None
    if prediction == [1]:
        result = "This team is going to qualify to the Playoff"
    else:
        result = "This team is not going to qualify to the Playoff"

    return render_template('index.html', prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)

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
    
    
    func = lambda prediction: "Qualify" if prediction==[1] else "Not qualify"
    prediction = model.predict(final_features)

    return render_template('index.html', prediction_text=func


if __name__ == "__main__":
    app.run(debug=True)

# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 02:15:35 2020

@author: Mohammed Shareef
"""

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)
pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return 'welcome sharif bhai'

@app.route('/predict',methods=["Get"])
def predict_note_authentication():
    
    """Let's Authenticate the Banks Note 
    Please enter the data of currency notes which were extracted from images.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
    """
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    
    if prediction==0:
        output1='It is a genuine currency note'
        return ' '+ str(output1)
    else:
        output2='It is a fake currency note'
        return ' '+ str(output2)
    
    
#    return 'The predicted value is'+ str(prediction)

if __name__=='__main__':
    app.run()

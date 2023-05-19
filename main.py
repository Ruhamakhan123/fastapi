import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

import pickle
import sklearn
from BankNote import BankNote
app = FastAPI()


pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.get('/')
def index():
    return "Hello"

@app.get('/{name}')
def get_name(name: str):
    return {'message': f'Hello, {name}'}


@app.get('/gmail')
def get_name():
    return {'message'}

@app.post('/predict')
def predict_note(data: BankNote):
   variance = data.variance
   skewness = data.skewness
   curtosis = data.curtosis
   entropy = data.entropy
   prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
   if prediction[0] > 0.5:
       result = 'Fake Note'
   else:
       result = 'Bank Note'
   return {'prediction': result}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

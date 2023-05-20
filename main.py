import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
import numpy as np

import pickle
import sklearn
from BankNote import BankNote
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

html = f"""
<!DOCTYPE html>
<html>
    <head>
        <title>FastAPI on Vercel</title>
        <link rel="icon" href="/static/favicon.ico" type="image/x-icon" />
    </head>
    <body>
        <div class="bg-gray-200 p-4 rounded-lg shadow-lg">
            <h1>Hello from FastAPI</h1>
            <ul>
                <li><a href="/docs">/docs</a></li>
                <li><a href="/redoc">/redoc</a></li>
            </ul>
            <p>Powered by <a href="https://vercel.com" target="_blank">Vercel</a></p>
        </div>
    </body>
</html>
"""


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

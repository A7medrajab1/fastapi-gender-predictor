# 1. Library imports
import uvicorn
from fastapi import FastAPI
from Gender import Gender  # Make sure this Pydantic model is defined
import numpy as np
import pickle
import pandas as pd

# 2. Create the app object
app = FastAPI()
pickle_in = open("Classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

# 3. Index route
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Name route
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To Fast API Session': f'{name}'}

# 5. Prediction route
@app.post('/predict')
def predict_gender(data: Gender):
    data = data.dict()
    Height = data['Height']
    Weight = data['Weight']
    
    prediction = classifier.predict([[Height, Weight]])
    prediction = "Male" if prediction[0] == 1 else "Female"
    
    return {'prediction': prediction}

# 6. Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=7002)

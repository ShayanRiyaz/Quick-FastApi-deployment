import uvicorn
import joblib
import pandas as pd 
from fastapi import FastAPI
from pydantic import BaseModel 
from data.predictions_handler import get_predictions
from loguru import logger

# initial App instance

app = FastAPI(title = 'Forged or Not Forged', version = '1.0',
            description = 'A Simple Neural Network that classified a banknote as either forged or not')


# Design incoming feature data
class Features(BaseModel):
    variance_of_wavelet: float
    skewness_of_wavelet: float
    curtosis_of_wavelet: float
    entropy_of_wavelet: float

# Initiate logging
log_format = "{time} | {level} | {message} | {file} | {line} | {function} | {exception}"
logger.add(sink = 'app/data/log_files/logs.log', format = log_format, level = "DEBUG", compression = 'zip')

# API root or home endpoint

@app.get('/')
@app.get('/home')
def read_home():
    """
    Home endopoint, can be used to test the availibility of the application.

    :return: Dict with key 'message' and value 'Fake or Not API live!'
    """

    logger.debug('User checked the root page')
    return {'message': 'Fake or Not API live!'}


# Prediction endpoint
@app.post('/predict')
@logger.catch() # Catch any unexpected breaks
def get_prediction(incoming_data: Features):
    """
    This function is the end point and serves us the predictions based on the values from a user and the saved model

    :param incoming_data: JSON with keys representing features and values representing the associated values
    :return: Dict with kyes "predicted_class" (class predicted ) and 'predicted_prob' (probability of predictions)
    """

    # retrieve incoming json data as a dictionary
    new_data = incoming_data.dict()
    logger.info('User sent some data for predictions')

    # Make predictions based on the incoming data and saved neural net
    preds = get_predictions(new_data)
    logger.debug('Predictions successfully generated for the user')

    # Return the predicted class and the predicted probability
    return {'predicted_class': round(float(preds.flatten())),'predicted_prob': float(preds.flatten())}

if __name__ == "__main__":
    # Run app with uvicorn with port and host specified. Host needed for docket port mapping
    uvicorn.run(app,port =8000,host = "0.0.0.0")

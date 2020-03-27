import tensorflow as tf 
import pandas as pd 
import numpy as np 
import joblib 
import os
from loguru import logger 

@logger.catch 
def get_predictions(data):
    
    """
    1. This function reshapes the incoming JSON data.
    2. Loads the object as a saved model.
    3. Returns predicted class and its probability.

    :param data: Dictionairy (Dict) with keys that represent 
    features and values that represented the respective value.

    :return: Dictionairy with keys 'predicted_class' and 'predicted_prob
    """

    # 1. Convert new data dictionairy (Dict) as a DataFrame
    # 2. Reshape columns to suit out model
    new_data = {k: [v] for k,v in data.items()}

    new_data_df = pd.DataFrame.from_dict(new_data)
    new_data_df = new_data_df[['variance_of_wavelet','skewness_of_wavelet',
                                'curtosis_of_wavelet', 'entropy_of_wavelet']]

    # Load saved standardising object
    scaler = joblib.load('app/data/scaler.joblib')
    logger.debug('Standardising object loaded 👍 ')

    #load saved kerasm model

    model = tf.keras.models.load_model('app/data/banknote_authentication_model.h5')
    logger.debug('Saved ANN model loaded successfully')

    # Scale new data using loaded object

    X = scaler.transform(new_data_df.values)
    logger.debug('Incoming data successfully standardised with saved object')

    # Make new predictions from the newley scaled data and return this prediction.

    preds = model.predict(X)

    return preds
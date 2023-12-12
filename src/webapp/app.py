import os
import json

from fastapi import FastAPI
from pydantic import BaseModel
from pydantic.fields import Field

import pickle

from loguru import logger


SENTIMENT_ANALYZER_MODEL_PATH = os.environ.get('SENTIMENT_ANALYZER_MODEL_PATH')


app = FastAPI(title='Sentiment Analysis API', api='v1', version='1.0.0')


class PredictInput(BaseModel):
    reviews: list[str] = Field(description='List of reviews',
                               example=['This movie is very good',
                                        'This movie is very bad'])


model = pickle.load(open(SENTIMENT_ANALYZER_MODEL_PATH + '/model.pkl', 'rb'))


@app.post('/predict',
          summary='Prediction of sentiment analysis on movies reviews')
def predict(input: PredictInput):
    '''
    Prediction of sentiment analysis on movies reviews
    @param input: list of reviews
    @return: list of sentiments
    '''
    logger.info('Predicting sentiments')
    try:
        y_pred = model.predict(input.reviews)
        logger.debug(f'Reviews: {input.reviews}')
        logger.debug(f'Predicted sentiments: {y_pred}')
        y_pred = ['negatif' if y == 0 else 'positif' for y in y_pred]
        return {
            'sentiments': y_pred
        }
    except Exception as e:
        logger.error(e)
        raise e


@app.get('/get_details',
         summary='Get details of the model used by the container')
def get_details():
    '''
    Get details of the model used by the container
    @return: details of the model
    '''
    with open(SENTIMENT_ANALYZER_MODEL_PATH + '/model_infos.json', 'r') as f:
        model_infos = f.read().replace('\'', '\"')
    model_infos = json.loads(model_infos)
    return model_infos


@app.get('/get_stage',
         summary='Get stage level of the model used by the container')
def get_stage():
    '''
    Get stage level of the model used by the container
    @return: stage level of the model
    '''
    with open(SENTIMENT_ANALYZER_MODEL_PATH + '/model_infos.json', 'r') as f:
        model_infos = f.read().replace('\'', '\"')
    model_infos = json.loads(model_infos)
    return {
        'stage': model_infos['_current_stage']
    }

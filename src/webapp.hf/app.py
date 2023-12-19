import os
import json

from fastapi import FastAPI
from pydantic import BaseModel
from pydantic.fields import Field

import pickle

import pymongo

from loguru import logger


SENTIMENT_ANALYZER_MODEL_PATH = os.environ.get('SENTIMENT_ANALYZER_MODEL_PATH')
URL_MONGO = os.environ.get('URL_MONGO')


client = pymongo.MongoClient(URL_MONGO)
db = client["sentiment_analyzer"]
collection = db["predictions"]


app = FastAPI(title='Sentiment Analysis API', api='v1', version='1.0.0')


class PredictInput(BaseModel):
    reviews: list[str] = Field(description='List of reviews',
                               example=['This movie is very good',
                                        'This movie is very bad'])


model = pickle.load(open('/model/model.pkl', 'rb'))


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

        i = 0
        for e in input.reviews:
            collection.insert_one({
                'review': e,
                'sentiment': str(y_pred[i])
            })
            i += 1

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


@app.get('/history',
         summary='Get the n last elements of the history of mongodb')
def get_history(n: int):
    '''
    Get the n last elements of the history of mongodb
    @param n: number of last elements
    @return: list of the n last elements of the history of mongodb
    '''
    logger.info('Getting history')
    try:
        history = collection.find().sort('_id', -1).limit(n)
        logger.debug(f'History: {history}')
        results = []
        for e in history:
            results.append({
                'review': e['review'],
                'sentiment': e['sentiment']
            })
        return {
            'history': results
        }
    except Exception as e:
        logger.error(e)
        raise e

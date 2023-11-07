# MLOPS project
Nicolas Deronsart M2ML


This project consists in the development of a MLOPS pipeline for the deployment of a machine learning model. The model aims to predict the polarity of a movie review in French. 

## Data

The data used to train and evaluate the models comes from the allocine website.
The dataset used is composed of three parts located in the data folder:
- train.csv: the training set, 160000 rows × 3 columns.
- test.csv: the test set, 20000 rows × 3 columns.
- valid.csv: the validation set, 20000 rows × 3 columns.

## Installation
To install and use the project, you need to install the python dependencies. To do so, you can use the following command:
```pip install -r requirements.txt```.

## Model
The model is currently a LogisticRegression using a TFIDF-vectorizer, both trained on the training set.

## Pipeline

TODO

## Project structure
- Notebooks folder containing the notebooks used to develop the project.
    - **exploratory_analysis.ipynb** : notebook used to explore the data to discover it.
    - **model_design.ipynb** : notebook used to preprocess the data and train a first model.
- Data folder containing the data used to train and evaluate the model.



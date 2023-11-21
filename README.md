# MLOPS project
Nicolas Deronsart M2ML


This project consists in the development of a MLOPS pipeline for the deployment of a machine learning model. The model aims to predict the polarity of a movie review in French. 

## Data

The data used to train and evaluate the models comes from the [AlloCiné](https://www.allocine.fr) website.
The dataset used is composed of three parts located in the data folder:
- train.csv: the training set, 160000 rows × 3 columns.
- valid.csv: the validation set, 20000 rows × 3 columns.
- test.csv: the test set, 20000 rows × 3 columns.
The three columns are:
- film-url: the url of the movie on the website, this information is not used in the project.
- review: the review of the movie.
- polarity: the polarity of the review, 0 for negative and 1 for positive.

## Installation
To install and use the project, you need to configure a conda environment using the requirements.txt file. To do so, you can use the following command:
```conda create --name mlops --file requirements.txt``````

## Model
The model is currently a LogisticRegression using a TFIDF-vectorizer, both trained on the training set.

## Pipeline
The pipeline use a TF-IDF vectorizer with the french stopwords, and a LogisticRegression model. The pipeline is trained on the training set and evaluated on the validation set. The pipeline is then saved in a pickle file.

## MLFlow
An MLFlow server is used to track the experiments and the models. The server is launched using the following command:
```mlflow server```. The runs and the models can be seen on the MLFlow UI at the following address http://127.0.0.1:5000 in the *model_design* experiment.

## Project structure
```
.
├── data
│   ├── test.csv
│   ├── train.csv
│   └── valide.csv
├── notebooks
│   ├── exploratory_analysis.ipynb
│   ├── model_design_2.ipynb
│   ├── model_design_3.ipynb
│   └── model_design.ipynb
├── README.md
└── requirements.txt
```

- The folder Data contains the data used to train, evaluate and test the model.

- The folder notebooks contains the notebooks used to develop the project.
    - **exploratory_analysis.ipynb** : notebook used to explore the data to discover it.
    - **model_design.ipynb** : notebook used to preprocess the data and train a first model.
    - **model_design_2.ipynb** : notebook used to preprocess the data and train models using MLFlow to track the experiments.
    - **model_design_3.ipynb** : notebook used to preprocess the data and train a first model MLFlow to track the experiments with an experiment function.

- The file requirements.txt containing the python dependencies of the conda environment. To install these versions you have to type the following command in your conda environment: ```pip install -r requirements.txt```.


```
TEST_MODEL_NAME='Sentiment analysis pipeline with tfidf and logistic regression' TEST_MODEL_VERSION='2' TEST_FILE='data/test.csv' pytest tests/
```

```
pip install -e .
```

```
predict --input_file 'data/test.csv' --output_file 'data/predictions.csv' --model_name 'Sentiment analysis pipeline with tfidf and logistic regression' --model_version '2' 
```

```
promote --model_name 'Sentiment analysis pipeline with tfidf and logistic regression' --model_version '2' --status 'Staging'
```

```
```

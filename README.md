# MLOPS project
Nicolas Deronsart M2ML


This project consists in the development of a MLOPS pipeline for the deployment of a machine learning model. The model aims to predict the polarity of a movie review in French.  
The final product is a web application that allows the user to enter a review and get the polarity of the review. 

## Data

The data used to train and evaluate the models comes from the [AlloCiné](https://www.allocine.fr) website.
The dataset used is composed of three parts located in the data folder:
- train.csv: the training set, 160000 rows × 3 columns.
- valid.csv: the validation set, 20000 rows × 3 columns.
- test.csv: the test set, 20000 rows × 3 columns.
The three columns are:
- film-url: the url of the movie on the website, this information is not used in this project.
- review: the review of the movie.
- polarity: the polarity of the review, 0 for negative and 1 for positive.

## Environment setup
To install and use the project, you need to configure a conda environment using the requirements.txt file. To do so, you can use the following command:
```
conda create --name mlops --file requirements.txt
```

## Model
The model used to predict the polarity of a review is a pipeline using a TF-IDF vectorizer with the french stopwords, and a LogisticRegression model. The pipeline is trained on the training set and evaluated on the validation set. The pipeline is then saved in an MLFlow server.

## MLFlow
An MLFlow server is used to track the experiments and the models. The server is launched using the following command from your MLFlow folder:
```
mlflow server
```
The runs and the models can be seen on the MLFlow UI at the following address http://127.0.0.1:5000 in the *model_design* experiment.

## Application
The application is composed of two parts:
- An API that allows the user to make predictions on a review.
- A frontend app that allows the user to enter a review and get the polarity of the review. It can also display the n lasts reviews entered in the model and their predicted polarity.

### Webapp
The webapp is a FastAPI application that allows the user to make predictions on a review. It can be used either locally or in a docker container. The MLFlow server must be running to use the webapp.

#### Local
If you want to use the webapp you can start the API able to make predictions using the following commands from the root folder.
```
python src/webapp/get_mlflow_model.py --mlflow_server_uri 'http://127.0.0.1:5000' --model_name 'Sentiment analysis pipeline with tfidf and logistic regression' --model_version '2' --target_path $SENTIMENT_ANALYZER_MODEL_PATH
```
With *--model_name* the name of the model to load, *--model_version* the version of the model to load, *--mlflow_server_uri* the uri of the MLFlow server and *--target_path* the path where to save the model.

Then you can start the API using the command:
```
command uvicorn src.webapp.app:app --host 0.0.0.0 --reload
```

The API has its docs at the url http://127.0.0.1:8000/docs where you can try the API.

#### Docker container
To use the API in a docker container, you can build the image from the terminal in the src/webapp folder with:
```
docker build -t mlops-webapp:1.0-model-v2 --build-arg MLFLOW_SERVER_URI=http://host.docker.internal:5000 --build-arg MODEL_NAME='Sentiment analysis pipeline with tfidf and logistic regression' --build-arg MODEL_VERSION='2' .
```
With *MLFLOW_SERVER_URI* the uri of the MLFlow server, *MODEL_NAME* the name of the model to load and *MODEL_VERSION* the version of the model to load.

And then run the docker container with:
```
docker run -p 8080:8000 --name mlops-webapp mlops-webapp:1.0-model-v2
```
The docs, where you can also try the requests, is at the url http://127.0.0.1:8080/docs.

Once the container has been created you can start and stop it with the following commands:
```
docker start mlops-webapp
docker stop mlops-webapp
```

Another version of the webapp is webapp.hf which loads a HuggingFace model instead of a model saved in MLFlow. To use this version you can build the image from the terminal in the src/webapp.hf folder with for example:
```
docker build -t mlops-webapp-hf:1.0-model-v5 --build-arg MODEL_NAME='sentiment-analyzer' --build-arg MODEL_VERSION='5' --build-arg HF_ID='haltux' .
```
And then run the docker container with:
```
docker run -p 8080:80 --name mlops-webapp-hf mlops-webapp-hf:1.0-model-v5
```

### Frontend app
The frontend app is a streamlit web application that allows the user to enter a review and get the polarity of the review. It can also display the n lasts reviews entered in the model and their predicted polarity.

#### Local
The app can be started locally with the following command from the src/frontend folder:
```
streamlit run app.py
```
The app is then available at the url http://localhost:8501.

#### Docker container
To use the frontend app in a docker container, you can build the image from the terminal in the src/frontend folder with:
```
docker build -t mlops-frontend:1.0 --build-arg API_URL='http://127.0.0.1:8080' .
```
With *API_URL* the url of the webapp api.

Then you can run the docker container with:
```
docker run -p 9000:8501 --name mlops-frontend mlops-frontend:1.0
```

Once the container has been created you can start and stop it with the following commands:
```
docker start mlops-frontend
docker stop mlops-frontend
```

### Docker compose

A docker compose file is available in the root folder. It allows to start the webapp and the frontend app in two containers. It also starts a mongo database to store the reviews entered in the frontend app and the predictions done.

To build the containers you can use the following command:
```
PREDICTON_CONTAINER="mlops-webapp:1.0-model-v2" MONGO_INITDB_ROOT_USERNAME="admin" MONGO_INITDB_ROOT_PASSWORD="admin" docker-compose build
```
With *PREDICTON_CONTAINER* the name of the webapp container, *MONGO_INITDB_ROOT_USERNAME* the username of the mongo database and *MONGO_INITDB_ROOT_PASSWORD* the password of the mongo database.

And then you can start the containers with:
```
PREDICTON_CONTAINER="mlops-webapp:1.0-model-v2" MONGO_INITDB_ROOT_USERNAME="admin" MONGO_INITDB_ROOT_PASSWORD="admin" docker-compose up
```
The *PREDICTON_CONTAINER*, *MONGO_INITDB_ROOT_USERNAME* and *MONGO_INITDB_ROOT_PASSWORD* arguments are the same as for the build command.

When the containers are started, the frontend app is available at the url http://127.0.0.1:9000.

If you want to load the version of the project that is in Docker Hub you have to create a docker-compose.yml file in your computer and execute it with the following commands :
```
WEBAPP_IMAGE=nderonsart/webapp-hf:1 WEBAPP_IMAGE_VERSION=1 FRONTEND_IMAGE=nderonsart/frontend:1 docker-compose build

WEBAPP_IMAGE=nderonsart/webapp-hf:1 WEBAPP_IMAGE_VERSION=1 FRONTEND_IMAGE=nderonsart/frontend:1 docker-compose up
```

## Project structure
```
.
├── .github
│   └── workflows
│       └── docker-build.yml
├── data
│   ├── test.csv
│   ├── train.csv
│   └── valide.csv
├── notebooks
│   ├── exploratory_analysis.ipynb
│   ├── model_design_2.ipynb
│   ├── model_design_3.ipynb
│   └── model_design.ipynb
├── src
│   ├── frontend
│   │   ├── app.py
│   │   └── Dockerfile
│   ├── sentiment_analyzer
│   │   ├── __init__.py
│   │   ├── hf_export.py
│   │   ├── model_manager.py
│   │   ├── predict.py
│   │   └── promote.py
│   ├── webapp
│   │   ├── app.py
│   │   ├── Dockerfile
│   │   └── get_mlflow_model.py
│   └── webapp.hf
│       ├── app.py
│       ├── Dockerfile
│       └── get_mlflow_model.py
├── tests
│   └── test_model.py
├── .gitignore
├── docker-compose.yml
├── Makefile
├── README.md
├── requirements.in
├── requirements.txt
└── setup.py
```

- The folder .github contains the github actions used to build the docker images of the application and push them to Docker Hub when a new version is pushed to the main branch.

- The folder Data contains the data used to train, evaluate and test the model.

- The folder notebooks contains the notebooks used to develop the project.
    - **exploratory_analysis.ipynb**: notebook used to explore the data to discover it.
    - **model_design.ipynb**: notebook used to preprocess the data and train a first model.
    - **model_design_2.ipynb**: notebook used to preprocess the data and train models using MLFlow to track the experiments.
    - **model_design_3.ipynb**: notebook used to preprocess the data and train a first model MLFlow to track the experiments with an experiment function.

- The folder src contains the source code of the project. It contains a sentiment_analyzer package with the following files:
    - **__init__.py**: the file used to make the package a python package.
    - **hf_export.py**: the file containing the function to export a model from the HuggingFace library to MLFlow. Once the package has been installed, this function can be called from the command line with the command ```hf_export```. For example:
    ```
    hf_export --model_name 'Sentiment analysis pipeline with tfidf and logistic regression' --model_version '2' --data_file 'data/train.csv' --hf_id 'your_huggingface_id' --hf_token 'your_huggingface_token' 
    ```
    - **model_manager.py**: the file containing the class ModelManager that can load a model from the MLFlow server and make predictions.
    - **predict.py**: file containing the function to make predictions on a dataset and saving the file in the output_file. Once the package has been installed, this function can be called from the command line with the command ```predict```. For example:
    ```
    predict --input_file 'data/test.csv' --output_file 'data/predictions.csv' --model_name 'Sentiment analysis pipeline with tfidf and logistic regression' --model_version '2' 
    ```
    - **promote.py**: file containing the function to promote a model to a specific stage. 
    The status can be either "Staging", "Production" or "Archived".
    If the chosen status is "Production", the model will be promoted if the tests in the tests folder are passed. 
    Once the package has been installed, this function can be called from the command line with the command ```promote```. For example:
    ```
    promote --model_name 'Sentiment analysis pipeline with tfidf and logistic regression' --model_version '2' --status 'Staging'
    ```
    In the src folder we already have the application source code

- The folder tests contains the tests of the project. It only contains the file test_model.py which run tests on a model saved in MLFlow. It checks if the model returns the expected type of output and that the accuracy score on the test set is greater than a threshold.
To run the tests you have to use the following command with the environment variables TEST_MODEL_NAME, TEST_MODEL_VERSION and TEST_FILE:
```
TEST_MODEL_NAME='Sentiment analysis pipeline with tfidf and logistic regression' TEST_MODEL_VERSION='2' TEST_FILE='data/test.csv' pytest tests/
```

- The file .gitignore contains the files and folders that are ignored by git.

- The file Makefile contains the commands to run the project to create the sentiment_analyzer package. It contains the following commands:
    - **build_dev_requirements**: command to build the requirements.txt file from the requirements.in file.
    - **install_dev_requirements**: command to install the requirements.txt file in the conda environment.
    - **install**: command to install the packages in the src folder in the conda environment.

- The file docker-compose.yml contains the configuration to run the webapp, the frontend app and the mongo database in docker containers.

- The file requirements.txt containing the python dependencies of the conda environment. To install these versions you have to type the following command in your conda environment: ```pip install -r requirements.txt```.

- The file setup.py contains the information to install the package in the conda environment. To install the package in the conda environment you have to type the following command: ```pip install -e .```.

FROM python:3.9-slim

ARG MLFLOW_SERVER_URI
ARG MODEL_NAME
ARG MODEL_VERSION

ENV MLFLOW_SERVER_URI=${MLFLOW_SERVER_URI}
ENV MODEL_NAME=${MODEL_NAME}
ENV MODEL_VERSION=${MODEL_VERSION}
ENV SENTIMENT_ANALYZER_MODEL_PATH=/model

WORKDIR /webapp

RUN pip install mlflow==2.6.0 fastapi==0.104.1 uvicorn==0.24.0.post1 loguru==0.7.2 pymongo==4.6.1

COPY ./app.py .
COPY ./get_mlflow_model.py .

RUN python get_mlflow_model.py --mlflow_server_uri ${MLFLOW_SERVER_URI} --model_name "${MODEL_NAME}" --model_version ${MODEL_VERSION} --target_path ${SENTIMENT_ANALYZER_MODEL_PATH}
RUN pip install -r /model/requirements.txt

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

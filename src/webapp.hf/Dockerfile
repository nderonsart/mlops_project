FROM python:3.9-slim

WORKDIR /webapp
RUN pip install mlflow fastapi uvicorn loguru pymongo skops

ARG HF_ID
ARG MODEL_NAME
ARG MODEL_VERSION

RUN python -c "from skops import hub_utils; hub_utils.download(repo_id='${HF_ID}/${MODEL_NAME}-${MODEL_VERSION}', dst='/model')"
COPY app.py ./
EXPOSE 80
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]

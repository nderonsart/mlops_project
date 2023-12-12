FROM python:3.9-slim

ARG API_URL

ENV API_URL=${API_URL}

WORKDIR /frontend

RUN pip install streamlit==1.29.0 

COPY ./app.py .

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]

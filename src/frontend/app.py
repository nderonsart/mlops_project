import streamlit as st
import requests

import os


API_URL = os.environ.get('API_URL')


st.title('Sentiment Analyzer')
text_area = st.text_area(label='Enter your rewiew here', height=20)
y = {
  "reviews": [
    text_area
  ]
}
button = st.button(label='Predict')
if button:
    response = requests.post(API_URL + '/predict', json=y)
    if response.status_code == 200:
        st.success(f'Sentiment: {response.json()["sentiments"][0]}')
    else:
        st.error('Error: Something went wrong')

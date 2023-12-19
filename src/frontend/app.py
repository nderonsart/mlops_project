import streamlit as st
import requests

import os


API_URL = os.environ.get('API_URL')


st.title('Sentiment Analyzer')

tabs = ['Prediction', 'History']

st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", tabs)

if selection == 'Prediction':
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

elif selection == 'History':
    number = st.number_input(label='Number of predictions to display',
                             min_value=1, max_value=100, value=1)
    response = requests.get(API_URL + '/history', params={'n': number})
    data = response.json()["history"]
    st.write('---')
    for e in data:
        st.write(f'Review: {e["review"]}')
        st.write(f'Sentiment: {e["sentiment"]}')
        st.write('---')

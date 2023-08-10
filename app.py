# -*- coding: utf-8 -*-
"""

"""



import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load

st.title('Model Deployment: Fake News Classifier')

st.sidebar.header('User Input Parameters')

def user_input_features():
    Time = st.sidebar.number_input("Insert Time")
    V1 = st.sidebar.number_input("V1")
    V2 = st.sidebar.number_input("V2")
    V3 = st.sidebar.number_input("V3")
    V4 = st.sidebar.number_input("V4")
    V5 = st.sidebar.number_input("V5")
    V6 = st.sidebar.number_input("V6")
    V7 = st.sidebar.number_input("V7")
    V8 = st.sidebar.number_input("V8")
    V9 = st.sidebar.number_input("V9")
    V10 = st.sidebar.number_input("V10")
    V11= st.sidebar.number_input("V11")
    V12 = st.sidebar.number_input("V12")
    V13= st.sidebar.number_input("V13")
    V14= st.sidebar.number_input("V14")
    V15= st.sidebar.number_input("V15")
    V16= st.sidebar.number_input("V16")
    V17= st.sidebar.number_input("V17")
    V18= st.sidebar.number_input("V18")
    V19= st.sidebar.number_input("V19")
    V20= st.sidebar.number_input("V20")
    V21= st.sidebar.number_input("V21")
    V22= st.sidebar.number_input("V22")
    V23 = st.sidebar.number_input("V23")
    V24 = st.sidebar.number_input("V24")
    V25 = st.sidebar.number_input("V25")
    V26 = st.sidebar.number_input("V26")
    V27 = st.sidebar.number_input("V27")
    V28 = st.sidebar.number_input("V28")
    Amount=st.sidebar.number_input("Amount")
    
    data = {'Time':Time,
            'V1':V1,
            'V2':V2,
            'V3':V3,
            'V4':V4,
            'V5':V5,
            'V6':V6,
            'V7':V7,
            'V8':V8,
            'V9':V9,
            'V10':V10,
            'V11':V11,
            'V12':V12,
            'V13':V13,
            'V14':V14,
            'V15':V15,
            'V16':V16,
            'V17':V17,
            'V18':V18,
            'V19':V19,
            'V20':V20,
            'V21':V21,
            'V22':V22,
            'V23':V23,
            'V24':V24,
            'V25':V25,
            'V26':V26,
            'V27':V27,
            'V28':V28,
            'Amount':Amount
         }
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)


# load the model from disk
loaded_model = load(open('deployment.pkl', 'rb'))

prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)

st.subheader('Fraud or Legit')
st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')
st.write(prediction)




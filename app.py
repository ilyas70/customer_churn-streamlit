import pickle
import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder

st.title('Churn Prediction')
st.subheader('Please enter your data:')

# input fields for the user
tenure = st.number_input("tenure",min_value = 0, max_value = 80, value = 45)

InternetService = st.selectbox("InternetService",["No","DSL","Fiber optic"])

OnlineSecurity= st.selectbox("OnlineSecurity",["No","Yes","No internet service"])

OnlineBackup = st.selectbox("OnlineBackup",["No","Yes","No internet service"])

DeviceProtection = st.selectbox("DeviceProtection",["No","Yes","No internet service"])

TechSupport = st.selectbox("TechSupport",["No","Yes","No internet service"])

Contract = st.selectbox("Contract",["Month-to-month","One year","Two year"])

PaymentMethod = st.selectbox("PaymentMethod",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])

# prepare the input data as a dictoary
input_data={
    "tenure":tenure,
    "InternetService":InternetService,
    "OnlineSecurity":OnlineSecurity,
    "OnlineBackup":OnlineBackup,
    "DeviceProtection":DeviceProtection,
    "TechSupport":TechSupport,
    "Contract":Contract,
    "PaymentMethod":PaymentMethod
    }

df = pd.read_csv('features.csv')
columns_list = df.columns.to_list()
# convert input data to dataframe
new_data = pd.DataFrame([input_data])

# load saved labelencoders
lmh={'Yes': 1, 'No': 0,'No internet service':2,'DSL':1,'Fiber optic':2
    ,'Month-to-month':0, 'One year':1, 'Two year':2,'Electronic check':0, 'Mailed check':1,
    'Bank transfer (automatic)':2,'Credit card (automatic)':3}

new_data['InternetService'] = new_data['InternetService'].map(lmh)
new_data['OnlineSecurity'] = new_data['OnlineSecurity'].map(lmh)
new_data['OnlineBackup'] = new_data['OnlineBackup'].map(lmh)
new_data['DeviceProtection'] = new_data['DeviceProtection'].map(lmh)
new_data['TechSupport'] = new_data['TechSupport'].map(lmh)
new_data['Contract'] = new_data['Contract'].map(lmh)
new_data['PaymentMethod'] = new_data['PaymentMethod'].map(lmh)

# Reindex to match the original column order
new_data = new_data.reindex(columns=columns_list, fill_value=0)


# Load the RandomForest model
with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Make predictions
prediction = loaded_model.predict(new_data)

if st.button('enter'):
    if prediction[0] == 1:
        st.error("Yes customer churned")
    else:
        st.success("No customer haven't churned")
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

#Loading trained model
model=tf.keras.models.load_model('regression_model.h5')

##Loading encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender=pickle.load(file)

with open('one_hot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder_geo=pickle.load(file)

with open('reg_scaler.pkl', 'rb') as file:
    scaler=pickle.load(file)   

##streamlit app
st.title('Estimated Salary Prediction')

##user input
geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Exited', [0,1])
tenure = st.slider('Tenure', 0, 20)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

#Preparing input data
input_data= pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

#One-hot encoding Geography
geo_encoded=one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

##combining onehot encoded columns with input data
input_data=pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

##scale the input data
input_data_scaled=scaler.transform(input_data)

##Churn prediction
prediction = model.predict(input_data_scaled)
predicted_salary= prediction[0][0]

st.write(f"The Estimated Salary is: eur {predicted_salary:.2f}")

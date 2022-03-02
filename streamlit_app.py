# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 18:58:39 2022

@author: adeep
"""

import numpy as np
import streamlit as st
import pickle


loaded_model = pickle.load(open("C:/Users/adeep/Downloads/trained_model.sav",'rb'))


def diabetes_prediction(input_data):
    
    #input_data = (5,166,72,19,175,25.8,0.587,51)
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  

def main():
    
    st.title('Diabetes Prediction -- SVM')
    
     
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose level')    
    BloodPressure = st.text_input('BloodPressure value')
    SkinThickness = st.text_input('SkinThickness')    
    Insulin = st.text_input('Insulin Value')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction Value')
    Age = st.text_input('Age of the person')
    
    
    # Code for prediction
    
    diagnosis = ''
    
    # Creating a button for prediction
    
    if st.button('Diabetes Test result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose, BloodPressure, SkinThickness,Insulin, BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)
    
if __name__ == "__main__":
    main()

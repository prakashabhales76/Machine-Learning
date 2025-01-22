import pandas as pd
import streamlit as st
from pycaret.regression import load_model, predict_model

#STEP1 : IMPORT THE TRAINED MODEL PIPELINE
# import the trained model
model=load_model('Final_model')


# STEP2: GET NEW DATA FOR PREDICTION FROM THE FRONT END
st.title("App to predict the house prices")
crim=st.slider('CRIM', 0.001, 100.0)
zn=st.slider('ZN',0.001, 100.0)
indus=st.slider('INDUS', 0.5, 28.0)
chas=st.selectbox('CHAS', [0,1])
nox=st.slider('NOX',0.4, 0.8)
rm=st.slider('RM', 3,9)
age=st.slider('AGE',2.9, 100.0)
dis=st.slider('DIS', 1.1, 12.1)
rad=st.slider('RAD',0,28)
tax=st.slider('TAX',187,800)
ptratio=st.slider('PRATIO',12.6, 22.0)
b=st.slider('B',0.3, 397.0)
lstat=st.slider('LSTAT', 1.7, 40.0)

data={
    'CRIM':crim,
    'ZN':zn,
    'INDUS':indus, 
    'CHAS':chas,
    'NOX':nox,
    'RM':rm, 
    'AGE':age,
    'DIS':dis,
    'RAD':rad,
    'TAX':tax,
    'PTRATIO':ptratio,
    'B':b, 
    'LSTAT':lstat
}

input_data=pd.DataFrame([data])
#input_data=str(input_data)

# STEP3 : GET THE PREDICTION AND DISPLAY IT
if st.button("Predict"):
    prediction=predict_model(model, input_data)
    st.success("The predicted price of the house in 1000's of $ is as below")
    prediction['Label'][0]

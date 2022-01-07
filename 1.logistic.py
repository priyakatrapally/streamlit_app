# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:01:17 2020

"""

import pandas as pd
import streamlit as st 
from sklearn.linear_model import LogisticRegression

st.title('Model Deployment: Logistic Regression')

st.sidebar.header('User Input Parameters')

def user_input_features():
    CLMSEX = st.sidebar.selectbox('Gender',('female','male'))
    CLMINSUR = st.sidebar.selectbox('Insurance',('1','0'))
    SEATBELT = st.sidebar.selectbox('SeatBelt',('1','0'))
    CLMAGE = st.sidebar.number_input("Insert the Age")
    LOSS = st.sidebar.number_input("Insert Loss")
    
    if (CLMSEX=='female'):
        CLMSEX='1'
    else:
        CLMSEX='0' 
        
    data = {'CLMSEX':CLMSEX,
            'CLMINSUR':CLMINSUR,
            'SEATBELT':SEATBELT,
            'CLMAGE':CLMAGE,
            'LOSS':LOSS}
       
    
    features = pd.DataFrame(data,index = [1])
    return features 
    
df = user_input_features()
st.subheader('Claimant Details')
st.write(df)

claimants = pd.read_csv("claimants.csv")
claimants.drop(["CASENUM"],inplace=True,axis = 1)
claimants = claimants.dropna()

X = claimants.iloc[:,[1,2,3,4,5]]
Y = claimants.iloc[:,0]
clf = LogisticRegression()
clf.fit(X,Y)

prediction_proba = clf.predict_proba(df)

st.subheader('Predicted Result')
st.write('Will not hire an attorney' if prediction_proba[0][1] > 0.553 else 'will hire an attorney')

st.subheader('Prediction Probability')
st.write(prediction_proba)
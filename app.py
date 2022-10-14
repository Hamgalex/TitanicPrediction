import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# leer los csv's
gender_submission = pd.read_csv("datasets/gender_submission.csv",encoding='utf-8')
test = pd.read_csv("datasets/test.csv",encoding='utf-8')
train = pd.read_csv("datasets/train.csv",encoding='utf-8')

# mergear los csv's
test=pd.merge(test,gender_submission,on="PassengerId", how="left")


st.title("Titanic Survivor Analysis")
st.header("Train data")
st.dataframe(train)
st.header("Test data")
st.dataframe(test)

# contar cuantos sobrevivieron
num_sobrevivientes = test.Survived.value_counts().to_list()

# graficar
fig = plt.figure(figsize = (10, 5))
plt.bar(["Female","Male"], num_sobrevivientes, color=['pink','lightskyblue'])
plt.title("Survivors per Sex") 

st.header("Number of survivors of the titanic per Sex")
st.pyplot(fig)


import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier


gender_submission = pd.read_csv("datasets/gender_submission.csv",encoding='utf-8')
test = pd.read_csv("datasets/test.csv",encoding='utf-8')
train = pd.read_csv("datasets/train.csv",encoding='utf-8')

test=pd.merge(test,gender_submission,on="PassengerId",how="left")
train=pd.merge(train,gender_submission,on="PassengerId",how="left")

st.dataframe(gender_submission)

st.dataframe(train)

st.dataframe(test)


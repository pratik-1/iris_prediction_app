import pickle

import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Iris Flower Prediction App
 This app predicts the **Iris flower** type!
 
 The data is obtained from [iris dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris)
""")

st.sidebar.header('User Input Parameters')
with open('data/iris_example_set.csv') as f:
   st.sidebar.download_button('Example input CSV file', f)

uploaded_file = st.sidebar.file_uploader('Upload the input CSV file', type='csv')
if uploaded_file is not None:
    input_features = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
        sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
        petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
        petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
        data = {'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width}
        features = pd.DataFrame(data, index=[0])
        return features
    input_features = user_input_features()


iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df = pd.concat([input_features, X], axis=0, ignore_index=True)

# st.write(iris)
df=df[:1]

st.subheader('User Input parameters')
if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently showing example input parameters.')
    st.write(df)

load_clf = pickle.load(open('iris_clf.pckl','rb'))
prediction = load_clf.predict(df)
prediction_probs = load_clf.predict_proba(df)

st.subheader('Prediction')
prediction_target = np.array(['setosa', 'versicolor', 'virginica'])
st.write(prediction_target[prediction])

st.subheader('Prediction Probability')
st.write(prediction_probs)
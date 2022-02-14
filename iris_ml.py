import pickle

import numpy as np
import streamlit as st
import pandas as pd
from bokeh.layouts import gridplot
from sklearn import datasets


from bokeh.plotting import figure, column
from bokeh.models import ColumnDataSource
from bokeh.models import CategoricalColorMapper


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
y = iris['target']
df = pd.concat([input_features, X], axis=0, ignore_index=True)

iris1 = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
iris1['species1'] = iris['target_names'][y]

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


def draw_plots_layouts(df, input_features):
    source = ColumnDataSource(df)
    color_mapper = CategoricalColorMapper(factors=['setosa', 'versicolor', 'virginica'],
                                          palette=['red', 'blue', 'green'])

    p1 = figure(x_axis_label='Sepal Length', y_axis_label='Sepal Width', tools='box_select,lasso_select')
    p1.circle('sepal_length', 'sepal_width', size=8, source=source, legend_field='species1',
              color=dict(field='species1', transform=color_mapper))
    p1.circle(input_features['sepal_length'], input_features['sepal_width'], size=10, color='yellow', legend_label='input feature')

    p2 = figure(x_axis_label='Petal Length', y_axis_label='Petal Width', tools='box_select,lasso_select')
    p2.circle('petal_length', 'petal_width', size=8, source=source, legend_field='species1',
              color=dict(field='species1', transform=color_mapper))
    p2.circle(input_features['petal_length'], input_features['petal_width'], size=10, color='yellow',
              legend_label='input feature')
    p2.legend.location = 'bottom_right'

    p3 = figure(x_axis_label='Sepal Length', y_axis_label='Petal Length', tools='box_select,lasso_select')
    p3.circle('sepal_length', 'petal_length', size=8, source=source, legend_field='species1',
              color=dict(field='species1', transform=color_mapper))
    p3.circle(input_features['sepal_length'], input_features['petal_length'], size=10, color='yellow',
              legend_label='input feature')
    p3.legend.location = 'bottom_right'

    p4 = figure(x_axis_label='Sepal Width', y_axis_label='Petal Width', tools='box_select,lasso_select')
    p4.circle('sepal_width', 'petal_width', size=8, source=source, legend_field='species1',
              color=dict(field='species1', transform=color_mapper))
    p4.circle(input_features['sepal_width'], input_features['petal_width'], size=10, color='yellow',
              legend_label='input feature')
    p4.legend.location = 'center_right'
    p1.x_range = p3.x_range
    p2.x_range = p4.x_range
    p1.y_range = p2.y_range = p3.y_range = p4.y_range

    # Row Layout
    # layout=row(p1,p2)

    # Column Layout
    # layout = column(p1, p2, p3, p4)


    # Grid Layout
    row1 = [p1, p2]
    row2 = [p3, p4]
    layout = gridplot([row1, row2])

    # show(layout)
    st.bokeh_chart(layout, use_container_width=True)


st.subheader('Visualisations')
draw_plots_layouts(iris1, input_features)

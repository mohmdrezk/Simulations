from math import ceil
from sklearn.datasets import make_blobs
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from func.generate_data import generate_data
from func.generate_data_classif import generate_data_classif
from func.make_plotly_fig import make_plotly_fig, make_figure2
from func.classify import classify
from func.get_train_test import get_train_test
import streamlit as st

data_gen_technique = st.sidebar.selectbox(
      'How are data generated?',
     ('make_blobs', 'make_classification'))

num_runs = st.sidebar.slider('Number of Runs', 2, 30, 10)  # min, max, default
num_reps = st.sidebar.slider('Number of Repetitions per run', 2, 10, 5) 
num_categs = st.sidebar.slider('Number of Categories', 2, 3, 2) 
num_features = st.sidebar.slider('Number of features', 2, 40, 2) 

if data_gen_technique == 'make_classification':
    percent_informative = st.sidebar.slider('% of informative features', 1, 100, 100) 
    num_informative = ceil(percent_informative/100*num_features)
    class_separation = st.sidebar.slider('Class Separation', 0.1, 2.0, 1.0) 

elif data_gen_technique == 'make_blobs':
    cluster_std = st.sidebar.slider('Cluster std', 0.1, 20.0, 1.0)
    
# create streamlit app
st.title('Simulation - SVM')
st.markdown('This app is a demo of a SVM classifier on simulated data.')

if data_gen_technique == 'make_classification':
    df = generate_data_classif(num_runs,num_reps, num_categs, 
                               num_features, num_informative, class_separation)
    
    st.write('Number of features:', num_features)
    st.write('Number of informative features:', num_informative)
    
elif data_gen_technique == 'make_blobs':
    df = generate_data(num_runs,num_reps, num_categs, num_features, cluster_std=cluster_std)
 
 
fig_output = make_plotly_fig(df)
st.plotly_chart(fig_output, use_container_width=True, height=600)

if num_features>2:
    st.warning(f'NOTE: Figure displaying ONLY the first 2 dimensions of the data. \n REAL data has {num_features} dimensions.')
    # st.write('REAL data has {} dimensions.'.format(num_features))

print(df.head(5))

num_CVs = df['CV'].nunique()
accuracy = []
accuracy_avg = []

for iCV in range(1,num_CVs+1):

    accuracy_tmp=[]
    X_train, y_train, X_test, y_test = get_train_test(df,iCV,False)
    accuracy_tmp = classify(X_train, y_train, X_test, y_test)
    accuracy.append(accuracy_tmp)
    # print(f'Number training samples: {X_train.shape[0]}')
    # print(f'Number testing samples: {X_test.shape[0]}')
    # print(f'-- -- -- -- -- ')
    
    accuracy_tmp=[]
    X_train, y_train, X_test, y_test = get_train_test(df,iCV,True)
    accuracy_tmp = classify(X_train, y_train, X_test, y_test)
    accuracy_avg.append(accuracy_tmp)
    # print(f'Number training samples: {X_train.shape[0]}')
    # print(f'Number testing samples with Averaging: {X_test.shape[0]}')
    # print(f'-- -- -- -- -- ')
    
d = {'Accuracy': accuracy, 'Accuracy_avg': accuracy_avg}
x= pd.DataFrame(d)


st.plotly_chart(make_figure2(x))



st.header('Data used in classification')
st.dataframe(df)

st.header('Acurracy results without and with averaging')
st.dataframe(x)

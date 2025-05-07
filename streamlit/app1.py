import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    data = pd.read_csv('winequality-red.csv', sep=",")
    return data

st.header('Вывод данных и графиков')

data_load_state = st.text('Загрузка данных...')
data = load_data()
data_load_state.text('Данные загружены!')

st.subheader('Первые 5 значений')
st.write(data.head())

if st.checkbox('Показать все данные'):
    st.subheader('Данные')
    st.write(data)

st.subheader('Скрипичные диаграммы для числовых колонок')
for col in ['fixed acidity','volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
            'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']:
    fig1 = plt.figure(figsize=(10,6))
    ax = sns.violinplot(data=data, x='quality', y=col)
    ax.set_title(f'Распределение {col} по качеству вина')
    st.pyplot(fig1)
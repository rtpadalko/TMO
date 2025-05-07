import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache_data
def load_data():
    data = pd.read_csv('winequality-red.csv', sep=",")
    return data

@st.cache_data
def preprocess_data(data_in):
    data_out = data_in.copy()
    
    # Преобразование целевой переменной в бинарную
    data_out['quality'] = (data_out['quality'] >= 7).astype(int)

    # Масштабируем числовые признаки
    scale_cols = ['fixed acidity','volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                  'free sulfur dioxide', 'total sulfur dioxide', 'density', 
                  'pH', 'sulphates', 'alcohol']
    new_cols = []
    scaler = MinMaxScaler()
    scaler_data = scaler.fit_transform(data_out[scale_cols])
    for i, col in enumerate(scale_cols):
        new_col = col + '_scaled'
        new_cols.append(new_col)
        data_out[new_col] = scaler_data[:, i]

    X = data_out[new_cols]
    y = data_out['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=1, stratify=y)
    return X_train, X_test, y_train, y_test

# Функция отрисовки ROC-кривой
def draw_roc_curve(y_true, y_score, ax):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC AUC = %0.2f' % auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")

# Список моделей
models_list = ['LogR', 'KNN_5', 'SVC', 'Tree', 'RF', 'GB']
clas_models = {
    'LogR': LogisticRegression(),
    'KNN_5': KNeighborsClassifier(n_neighbors=5),
    'SVC': SVC(probability=True),
    'Tree': DecisionTreeClassifier(),
    'RF': RandomForestClassifier(),
    'GB': GradientBoostingClassifier()
}

def print_models(models_select, X_train, X_test, y_train, y_test):
    current_models_list = []
    roc_auc_list = []

    for model_name in models_select:
        model = clas_models[model_name]
        model.fit(X_train, y_train)

        Y_pred = model.predict(X_test)
        Y_pred_proba = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, Y_pred_proba)
        current_models_list.append(model_name)
        roc_auc_list.append(roc_auc)

        fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
        draw_roc_curve(y_test, Y_pred_proba, ax[0])

        cm = confusion_matrix(y_test, Y_pred, normalize='all')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(ax=ax[1], cmap=plt.cm.Blues)
        fig.suptitle(model_name)
        st.pyplot(fig)

    if roc_auc_list:
        temp_df = pd.DataFrame({'roc-auc': roc_auc_list}, index=current_models_list)
        st.bar_chart(temp_df)

# Интерфейс Streamlit
st.sidebar.header('Модели машинного обучения')
models_select = st.sidebar.multiselect('Выберите модели', models_list)

st.header('Оценка качества моделей')

data = load_data()
X_train, X_test, y_train, y_test = preprocess_data(data)

print_models(models_select, X_train, X_test, y_train, y_test)

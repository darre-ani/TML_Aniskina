import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

st.set_page_config(page_title="Анализ dc.csv", layout="wide")
st.title("Макет веб-приложения для анализа данных (dc.csv)")

# 1. Загрузка и подготовка данных
@st.cache_data
def load_data(filepath='dc.csv'):
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    # Создаём целевую переменную: 1 = цена выросла, 0 = цена упала
    df['target'] = (df['close_USD'] > df['open_USD']).astype(int)
    return df

df = load_data()

if df is not None and not df.empty:
    st.subheader("Загруженные данные (первые 5 строк)")
    st.dataframe(df.head())
    st.info(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов | Классы: {df['target'].value_counts().to_dict()}")
    
    # Выбор признаков (используем USD-метрики и объём)
    features = ['open_USD', 'high_USD', 'low_USD', 'close_USD', 'volume']
    X = df[features]
    y = df['target']
    
    # 2. Настройка модели и гиперпараметров
    st.subheader("Настройка модели")
    col_cfg1, col_cfg2 = st.columns([1, 2])
    
    with col_cfg1:
        model_name = st.selectbox("Выберите алгоритм",
                                 ["Logistic Regression", "Random Forest", "K-Nearest Neighbors", "SVM"])
        
        params = {}
        if model_name == "Logistic Regression":
            params["C"] = st.slider("C (регуляризация)", 0.01, 10.0, 1.0, 0.01)
            params["max_iter"] = 1000
        elif model_name == "Random Forest":
            params["n_estimators"] = st.slider("Деревья (n_estimators)", 10, 200, 50, 10)
            params["max_depth"] = st.slider("Максимальная глубина", 2, 20, 5, 1)
        elif model_name == "K-Nearest Neighbors":
            params["n_neighbors"] = st.slider("Количество соседей (k)", 1, 30, 5, 1)
        elif model_name == "SVM":
            params["C"] = st.slider("C (штраф за ошибку)", 0.1, 10.0, 1.0, 0.1)
            params["kernel"] = st.selectbox("Ядро", ["linear", "rbf"])
    
    with col_cfg2:
        st.markdown("**Логика целевой переменной:** `target = 1` если `close_USD > open_USD`, иначе `0`")
        st.markdown(f"**Признаки:** `{', '.join(features)}`")
    
    # 3. Кнопка обучения
    if st.button("Обучить модель", type="primary"):
        with st.spinner("Разделение данных, масштабирование и обучение..."):
            # Разделение (для финансовых рядов shuffle=False сохраняет хронологию, но для учебного макета оставим стандартное)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Масштабирование
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)
            
            # Инициализация модели
            if model_name == "Logistic Regression":
                model = LogisticRegression(C=params["C"], max_iter=params["max_iter"], random_state=42)
            elif model_name == "Random Forest":
                model = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=42)
            elif model_name == "K-Nearest Neighbors":
                model = KNeighborsClassifier(n_neighbors=params["n_neighbors"])
            elif model_name == "SVM":
                model = SVC(C=params["C"], kernel=params["kernel"], probability=True, random_state=42)
            
            # Обучение
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
            y_prob = model.predict_proba(X_test_sc)[:, 1] if hasattr(model, "predict_proba") else None
            
            # 4. Метрики
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            st.success(f"Обучение завершено! Модель: `{model_name}`")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Accuracy", f"{acc:.3f}")
            col_m2.metric("Precision", f"{prec:.3f}")
            col_m3.metric("Recall", f"{rec:.3f}")
            col_m4.metric("F1-Score", f"{f1:.3f}")
            
            # 5. Графики
            st.subheader("Визуализация результатов")
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                fig1, ax1 = plt.subplots(figsize=(5, 4))
                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax1)
                ax1.set_xlabel("Predicted")
                ax1.set_ylabel("True")
                st.pyplot(fig1)
            
            with col_g2:
                if y_prob is not None:
                    fig2, ax2 = plt.subplots(figsize=(5, 4))
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color='orange')
                    ax2.plot([0,1], [0,1], "k--", label="Random Guess")
                    ax2.set_xlabel("False Positive Rate")
                    ax2.set_ylabel("True Positive Rate")
                    ax2.legend()
                    st.pyplot(fig2)
                else:
                    st.info("ROC-кривая недоступна для выбранного ядра SVM.")
            
            # Распределение прогнозов
            fig3, ax3 = plt.subplots(figsize=(6, 3))
            pd.Series(y_pred).value_counts().sort_index().plot(kind='bar', ax=ax3, color=["#326b46", "#479fca"])
            ax3.set_xticklabels(['Падение (0)', 'Рост (1)'])
            ax3.set_ylabel("Количество наблюдений")
            ax3.set_title("Распределение предсказаний на тестовой выборке")
            st.pyplot(fig3)
else:
    st.error("Файл dc.csv не найден в папке проекта или имеет неверный формат.")
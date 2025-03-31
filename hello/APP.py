#python -m streamlit run app.py --server.port 8001

import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

API_URL = "http://127.0.0.1:8001"

#st.set_page_config(layout="wide")
st.title("Анализ данных и машинное обучение")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Об программе", "Датасет", "EDA", "Результаты моделей", "Визуализация моделей", "Предсказание"]
)

# --- TAB 6 ---
with tab6:
    st.header("Сделать предсказание по параметрам")
    gravity = st.number_input("Введите значение gravity", min_value=0.0, max_value=2.0, step=0.01, value=1.01)
    ph = st.number_input("Введите значение ph", min_value=0.0, max_value=14.0, step=0.01, value=6.5)

    if st.button("Предсказать"):
        try:
            payload = {"gravity": gravity, "ph": ph}
            response = requests.post(f"{API_URL}/predict", json=payload)
            result = response.json()

            st.subheader("Результаты предсказания моделей")
            for model_name, output in result.items():
                prob = output["probability"]
                if prob is not None:
                    st.write(f"**{model_name}**: Класс = {output['prediction']}, Вероятность = {prob:.2f}")
                else:
                    st.write(f"**{model_name}**: Класс = {output['prediction']} (вероятность не поддерживается)")
        except Exception as e:
            st.error(f"Ошибка предсказания: {e}")


# 1. Об программе
with tab1:
    st.header("Об программе")
    st.markdown("""
    Это Streamlit-приложение выполняет анализ данных и классификацию с использованием нескольких моделей:
    - Собственная реализация KNN (CustomKNN)
    - KNeighborsClassifier из sklearn
    - Логистическая регрессия
    - Support Vector Machine

    Данные: признаки `gravity`, `ph` и целевая переменная `target`.  
    Все вычисления (EDA, обучение, метрики, графики) выполняются через FastAPI.
    """)

# 2. Датасет
with tab2:
    st.header("Датасет")
    try:
        info = requests.get(f"{API_URL}/data_info").json()
        st.subheader("Общая информация:")
        st.write(f"Количество строк: {info['n_rows']}")
        st.write(f"Количество колонок: {info['n_cols']}")
        st.write(f"Объем в памяти: {info['memory_bytes']} байт")

        # Описание каждого поля
        st.write("### Описание полей:")
        description = {
        "gravity": "Гравитация (числовой признак)",
        "ph": "pH (числовой признак)",
        "osmo": "Осмос (целочисленный признак)",
        "cond": "Проводимость (числовой признак)",
        "urea": "Мочевина (целочисленный признак)",
        "calc": "Кальций (числовой признак)",
        "target": "Целевая переменная (бинарная классификация)",
        }
        for col, desc in description.items():
            st.write(f"**{col}**: {desc}")

        st.subheader("Первые 10 строк:")
        df = pd.DataFrame(info["head"])
        st.dataframe(df)
    except Exception as e:
        st.error(f"Ошибка получения информации о датасете: {e}")

# 3. EDA
with tab3:
    st.header("Exploratory Data Analysis (EDA)")

    st.write("""
        В этом разделе проводим анализ данных для выявления статистических характеристик, таких как минимальные, 
        максимальные значения, медиана, среднее, а также квартильные значения для числовых признаков.
        Также мы исследуем категориальные признаки, определяя их наиболее часто встречающиеся значения.
    """)

    try:
        eda = requests.get(f"{API_URL}/eda").json()
        st.subheader("Числовые признаки:")
        st.dataframe(pd.DataFrame(eda["numerical"]).set_index("feature"))

        st.subheader("Категориальные признаки:")
        if eda["categorical"]:
            for col, stats in eda["categorical"].items():
                st.write(f"**{col}**: most frequent = {stats['mode']}, count = {stats['frequency']}")
        else:
            st.info("Категориальных признаков нет.")
    except Exception as e:
        st.error(f"Ошибка получения EDA: {e}")

# 4. Результаты моделей
with tab4:
    st.header("Результаты моделей")
    try:
        metrics = pd.DataFrame(requests.get(f"{API_URL}/metrics").json()).set_index("name")
        st.subheader("Метрики моделей:")
        st.dataframe(metrics.style.highlight_max(axis=0, color='green'))

        st.subheader("ROC-кривые:")
        roc_data = requests.get(f"{API_URL}/roc").json()
        for model in roc_data:
            fpr = model["fpr"]
            tpr = model["tpr"]
            name = model["name"]
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(fpr, tpr, label=f"{name}")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax.set_title(f"ROC-кривая для модели: {name}")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            st.pyplot(fig, bbox_inches='tight', dpi=100)
    except Exception as e:
        st.error(f"Ошибка получения метрик или ROC: {e}")

# 5. Визуализация моделей
with tab5:
    st.header("Визуализация границ решений моделей")
    try:
        data = requests.get(f"{API_URL}/boundaries").json()
        X = np.array(data["X"])
        y = np.array(data["y"])
        for name, surface in data["boundaries"].items():
            xx = np.array(surface["xx"])
            yy = np.array(surface["yy"])
            Z = np.array(surface["Z"])
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.contourf(xx, yy, Z, cmap=ListedColormap(["#FFAAAA", "#AAFFAA"]), alpha=0.6)
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(["#FF0000", "#00AA00"]), edgecolor="k")
            ax.set_title(f"Границы решений модели: {name}")
            ax.set_xlabel("gravity (scaled)")
            ax.set_ylabel("ph (scaled)")
            st.pyplot(fig, bbox_inches='tight', dpi=200)
    except Exception as e:
        st.error(f"Ошибка визуализации границ: {e}")

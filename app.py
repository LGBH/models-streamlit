import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, accuracy_score, classification_report,
                             mean_squared_error, mean_absolute_error)
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

st.set_page_config(page_title="Modelos IA y FCM", layout="centered")
st.title("🤖 Clasificación con IA y FCM")

file = st.file_uploader("📂 Sube tu archivo Excel (.xlsx) con la variable 'C31' como etiqueta", type=["xlsx"])
modelo = st.selectbox("🔎 Selecciona el modelo a aplicar", [
    "Regresión Logística",
    "Red Neuronal Artificial",
    "Máquina de Apoyo Vectorial",
    "Mapa Cognitivo Difuso (Simulado)"
])

if file:
    df = pd.read_excel(file)
    if "C31" not in df.columns:
        st.error("La columna 'C31' no está en el archivo.")
        st.stop()

    st.subheader("📋 Vista previa del dataset")
    st.dataframe(df.head())

    X = df.drop(columns=["C31"])
    y = df["C31"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if modelo == "Regresión Logística":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif modelo == "Red Neuronal Artificial":
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif modelo == "Máquina de Apoyo Vectorial":
        model = SVC(kernel='linear')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    elif modelo == "Mapa Cognitivo Difuso (Simulado)":
        umbral = st.slider("🎚️ Ajusta el umbral de decisión FCM", min_value=0.0, max_value=100.0, value=30.0)
        y_pred = [1 if np.mean(row) > umbral else 0 for _, row in X_test.iterrows()]

        # Mostrar grafo simulado
        st.subheader("🧠 Grafo del FCM (Simulado)")
        G = nx.DiGraph()
        for col in X.columns:
            G.add_edge(col, "C31", weight=np.random.uniform(-1, 1))

        pos = nx.spring_layout(G)
        plt.figure(figsize=(8, 5))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1500, font_size=10)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()})
        st.pyplot(plt)

    # Métricas comunes
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.subheader(f"📊 Resultados del modelo - {modelo}")
    st.write(f"**Exactitud:** {accuracy * 100:.2f}%")

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusión")
    st.pyplot(fig)

    # Comparación real vs predicción
    st.subheader("📈 Comparación Real vs Predicción")
    df_res = pd.DataFrame({'Real': y_test.values, 'Predicción': y_pred})
    st.line_chart(df_res.reset_index(drop=True))

    # Reporte de clasificación
    st.subheader("📋 Reporte de Clasificación")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Errores (solo para RNA y SVM)
    if modelo in ["Red Neuronal Artificial", "Máquina de Apoyo Vectorial"]:
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        st.write(f"**Error cuadrático medio (MSE):** {mse:.4f}")
        st.write(f"**Error absoluto medio (MAE):** {mae:.4f}")

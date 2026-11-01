import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

st.set_page_config(page_title="Diagnóstico Breast Cancer", layout="wide")

# Cargar assets guardados y el dataset directamente desde UCI Repo
@st.cache_resource
def load_assets():
    model_rf = joblib.load('model_rf.pkl')
    model_lr = joblib.load('model_lr.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features.pkl')
    metrics_df = joblib.load('metrics_df.pkl')
    
    # Cargar dataset de UCI para la visualización en vivo
    breast_cancer = fetch_ucirepo(id=17) 
    X_df = breast_cancer.data.features 
    y_df = breast_cancer.data.targets
    
    df_full = pd.concat([X_df, y_df], axis=1)
    
    return {'Random Forest': model_rf, 'Regresión Logística': model_lr}, scaler, features, metrics_df, df_full

models, scaler, features, metrics_df, df_full = load_assets()

st.title("🩺 Predicción de Cáncer de Mama - Breast Cancer Wisconsin")
st.markdown("Ingrese los valores de los descriptores celulares y seleccione el modelo para predecir si la muestra corresponde a un tumor **Maligno** o **Benigno**.")

st.sidebar.header("⚙️ Configuración del Modelo")
selected_model_name = st.sidebar.selectbox(
    "Seleccione el Algoritmo de ML:",
    options=list(models.keys())
)
model = models[selected_model_name]

st.sidebar.header("📋 Parámetros de Entrada")
input_data = {}

feature_labels = {
    'radius1': 'Radio Promedio (mm)',
    'radius_mean': 'Radio Promedio (mm)',
    'texture1': 'Textura Promedio (Escala de Grises)',
    'texture_mean': 'Textura Promedio (Escala de Grises)',
    'perimeter1': 'Perímetro Promedio (mm)',
    'perimeter_mean': 'Perímetro Promedio (mm)',
    'area1': 'Área Promedio (mm²)',
    'area_mean': 'Área Promedio (mm²)',
    'smoothness1': 'Suavidad / Homogeneidad Promedio',
    'smoothness_mean': 'Suavidad / Homogeneidad Promedio'
}

default_ranges = [
    (6.0, 30.0, 14.12, "%.2f"),     # Radio
    (9.0, 40.0, 19.29, "%.2f"),     # Textura
    (43.0, 190.0, 91.96, "%.2f"),   # Perímetro
    (140.0, 2500.0, 654.88, "%.2f"),# Área
    (0.05, 0.25, 0.096, "%.5f")     # Suavidad
]

for idx, feat in enumerate(features):
    if idx < len(default_ranges):
        min_val, max_val, default_val, fmt = default_ranges[idx]
    else:
        min_val, max_val, default_val, fmt = (0.0, 100.0, 10.0, "%.2f")

    label_text = feature_labels.get(feat, feat.replace('_', ' ').title())

    input_data[feat] = st.sidebar.number_input(
        label=label_text,
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(default_val),
        format=fmt
    )

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(f"Resultados de la Predicción ({selected_model_name})")
    if st.button("Realizar Diagnóstico", type="primary"):
        df_input = pd.DataFrame([input_data])
        scaled_input = scaler.transform(df_input)

        prediction = model.predict(scaled_input)[0]
        proba = model.predict_proba(scaled_input)[0]

        if prediction == 1:
            st.error(f"⚠️ **Resultado: Maligno** (Probabilidad: {proba[1]*100:.2f}%)")
        else:
            st.success(f"✅ **Resultado: Benigno** (Probabilidad: {proba[0]*100:.2f}%)")

        st.write("---")
        st.metric(label="Probabilidad de Benigno", value=f"{proba[0]*100:.1f}%")
        st.metric(label="Probabilidad de Maligno", value=f"{proba[1]*100:.1f}%")

with col2:
    tab1, tab2, tab3 = st.tabs(["📊 Importancia / Coeficientes", "📈 Evaluación de Métricas", "📂 Vista del Dataset"])

    display_features = [feature_labels.get(f, f) for f in features]

    with tab1:
        st.subheader("Análisis de Características")
        if selected_model_name == 'Random Forest':
            st.markdown("**Importancia Relativa de Descriptores**")
            importances = model.feature_importances_
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=importances, y=display_features, palette="viridis", ax=ax)
            ax.set_xlabel("Importancia Relativa")
            st.pyplot(fig)
        else:
            st.markdown("**Impacto de Coeficientes en el Modelo**")
            coefs = model.coef_[0]
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=coefs, y=display_features, palette="coolwarm", ax=ax)
            ax.set_xlabel("Impacto en la Predicción")
            st.pyplot(fig)

    with tab2:
        st.subheader("Tabla Comparativa de Desempeño")
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='#d4edda'), use_container_width=True)

        st.markdown("**Métricas Clave del Modelo Seleccionado:**")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Accuracy", f"{metrics_df.loc[selected_model_name, 'Accuracy']*100:.1f}%")
        m_col2.metric("Precision", f"{metrics_df.loc[selected_model_name, 'Precision']*100:.1f}%")
        m_col3.metric("Recall", f"{metrics_df.loc[selected_model_name, 'Recall']*100:.1f}%")
        m_col4.metric("F1-Score", f"{metrics_df.loc[selected_model_name, 'F1-Score']*100:.1f}%")

    with tab3:
        st.subheader("Información General del Dataset UCI")
        
        # Conteo de datos
        total_filas, total_cols = df_full.shape
        target_col = df_full.columns[-1]
        conteo_diag = df_full[target_col].value_counts()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total de Muestras / Registros", total_filas)
        c2.metric("Casos Benignos (B)", conteo_diag.get('B', 0))
        c3.metric("Casos Malignos (M)", conteo_diag.get('M', 0))
        
        st.markdown("---")
        st.markdown("**Exploración de la Tabla Completa de Datos:**")
        st.dataframe(df_full, use_container_width=True)

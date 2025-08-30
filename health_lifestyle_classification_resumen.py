# =========================
# Librerías
# =========================
import os
import numpy as np
import pandas as pd
import streamlit as st
import scipy.sparse as sp
import matplotlib.pyplot as plt
from matplotlib import pyplot  
from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, KBinsDiscretizer, LabelEncoder
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2, RFECV
import shutil
import seaborn as sns
from scipy.stats import spearmanr, randint, uniform
import prince   # MCA
import mca      # Alternativa para MCA
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# =========================
# Configuración Streamlit
# =========================
st.set_page_config(page_title="Predicción de Riesgo de enfermedades - Resumen", layout="wide")
st.title("Predicción de Riesgo de enfermedades - Resumen")
sns.set_theme()

# =========================
# Carga de datos
# =========================
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    url = "https://drive.google.com/uc?export=download&id=1RdBTkEM9g-qKyeyqzE3WS6LSy1EhZwyO"
    data = pd.read_csv(url, low_memory=False)
    data.columns = data.columns.str.strip()
    return data

with st.spinner("Cargando datos..."):
    data = load_data()

# Eliminar columnas innecesarias 
cols_to_drop = ['survey_code', 'bmi_estimated', 'bmi_scaled', 'bmi_corrected']
present_to_drop = [c for c in cols_to_drop if c in data.columns]
if present_to_drop:
    data = data.drop(columns=present_to_drop)

st.info(f"Datos cargados: **{data.shape[0]:,} filas** × **{data.shape[1]:,} columnas**")

# ====================
# Filtros 
# ====================

with st.sidebar:
    st.header("Filtros")

    # Valores únicos seguros (sin NaN)
    def uniq(col):
        return sorted(data[col].dropna().astype(str).unique().tolist()) if col in data.columns else []

    # Edad
    if "age" in data.columns:
        age_series = pd.to_numeric(data["age"], errors="coerce")
        age_min = int(np.nanmin(age_series))
        age_max = int(np.nanmax(age_series))
        # Evitar que min==max rompa el slider
        if age_min == age_max:
            age_min = max(age_min - 1, 0)
            age_max = age_max + 1
        age_range = st.slider(
            "Rango de edad",
            min_value=age_min,
            max_value=age_max,
            value=(age_min, age_max),
            step=1,
            key="age_range"
        )
    else:
        age_range = None
        st.warning("No se encontró la columna 'age'.")

    gender_sel   = st.multiselect("Género", uniq("gender"), key="gender_sel")
    exercise_sel = st.multiselect("Tipo de ejercicio", uniq("exercise_type"), key="exercise_sel")
    edu_sel      = st.multiselect("Nivel educativo", uniq("education_level"), key="edu_sel")

# Aplicar filtros sobre copia y luego reemplazar `data`
data_filtered = data.copy()

if age_range and "age" in data_filtered.columns:
    data_filtered["age"] = pd.to_numeric(data_filtered["age"], errors="coerce")
    data_filtered = data_filtered[
        data_filtered["age"].between(age_range[0], age_range[1], inclusive="both")
    ]

if gender_sel and "gender" in data_filtered.columns:
    data_filtered = data_filtered[data_filtered["gender"].astype(str).isin(gender_sel)]

if exercise_sel and "exercise_type" in data_filtered.columns:
    data_filtered = data_filtered[data_filtered["exercise_type"].astype(str).isin(exercise_sel)]

if edu_sel and "education_level" in data_filtered.columns:
    data_filtered = data_filtered[data_filtered["education_level"].astype(str).isin(edu_sel)]

if data_filtered.empty:
    st.warning("No hay datos disponibles después de aplicar los filtros.")
    st.stop()

data = data_filtered.reset_index(drop=True)

st.info(
    f"Registros después de filtros: **{len(data):,}**"
    + (f" | Edad: {age_range[0]}–{age_range[1]}" if age_range else "")
    + (f" | Género: {', '.join(gender_sel)}" if gender_sel else "")
    + (f" | Ejercicio: {', '.join(exercise_sel)}" if exercise_sel else "")
    + (f" | Educación: {', '.join(edu_sel)}" if edu_sel else "")
)
# =========================
# Introductorio
# =========================
st.markdown("""
# Predicción de Riesgo de Enfermedades a partir de Hábitos de Vida

## Introducción
El cuidado de la salud está fuertemente influenciado por los hábitos diarios y las condiciones de vida de las personas. En este proyecto se busca aprovechar un conjunto de datos amplio y diverso que combina información demográfica, biométrica, conductual y psicológica con el fin de construir modelos de *machine learning* capaces de predecir la probabilidad de desarrollar una enfermedad. "[Abrir origen de los datos](https://www.kaggle.com/datasets/mahdimashayekhi/disease-risk-from-daily-habits)"

## El Conjunto de Datos
El dataset utilizado reúne información de **100,000 individuos**, describiendo un total de **40 variables explicativas y 1 variable objetivo**. La columna objetivo define el estado de salud del individuo en una clasificación binaria:
""")

# =========================
# Conteo de la variable objetivo
# =========================
st.subheader("Conteo de la variable objetivo (`target`)")
if 'target' in data.columns:
    # Excluir NaN por defecto
    counts = data['target'].dropna().value_counts()

    if counts.empty:
        st.info("No hay valores no nulos en `target` para graficar.")
    else:
        # Etiquetas y paleta viridis consistente por clase
        labels = counts.index.astype(str).tolist()
        cmap = plt.get_cmap("viridis")
        unique_sorted = sorted(set(labels))
        base_colors = cmap(np.linspace(0.2, 0.9, len(unique_sorted)))
        palette = dict(zip(unique_sorted, base_colors))
        colors_in_order = [palette[l] for l in labels]

        # --- Tamaño y nitidez fijos ---
        fig, ax = plt.subplots(figsize=(3, 3), dpi=120)  

        ax.bar(labels, counts.values, color=colors_in_order)
        ax.set_xlabel("Clase")
        ax.set_ylabel("Frecuencia")
        ax.set_title("Conteo de `target`")

        total = counts.sum()
        for i, v in enumerate(counts.values):
            ax.annotate(f"{v} ({v/total:.1%})", (i, v), ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        st.pyplot(fig, clear_figure=True, use_container_width=False) 
else:
    st.warning("No se encontró la columna `target` en el dataset.")



st.markdown("""
Se puede ver que las variables son:

## Variables:
- **Numéricas:** edad, IMC, presión arterial, colesterol, glucosa, horas de ejercicio, etc.
- **Categóricas:** género, estado civil, tipo de dieta, ocupación, etc.
""")

# =========================
# Nombres en español
# =========================
CUSTOM_ALIASES = {
    "age": "Edad", "gender": "Género", "height": "Talla (cm)", "weight": "Peso (kg)", "bmi": "IMC",
    "waist_size": "Perímetro de cintura", "blood_pressure": "Presión arterial", "heart_rate": "Frecuencia cardíaca",
    "cholesterol": "Colesterol", "glucose": "Glucosa", "insulin": "Insulina", "sleep_hours": "Horas de sueño",
    "sleep_quality": "Calidad del sueño", "work_hours": "Horas de trabajo", "physical_activity": "Actividad física",
    "daily_steps": "Pasos diarios", "calorie_intake": "Ingesta calórica", "sugar_intake": "Ingesta de azúcar",
    "alcohol_consumption": "Consumo de alcohol", "smoking_level": "Nivel de tabaquismo", "water_intake": "Consumo de agua",
    "screen_time": "Tiempo de pantalla", "stress_level": "Nivel de estrés", "mental_health_score": "Puntaje de salud mental",
    "mental_health_support": "Apoyo en salud mental", "education_level": "Nivel educativo", "job_type": "Tipo de trabajo",
    "occupation": "Ocupación", "income": "Ingresos", "diet_type": "Tipo de dieta", "exercise_type": "Tipo de ejercicio",
    "device_usage": "Uso de dispositivos", "healthcare_access": "Acceso a salud", "insurance": "Seguro de salud",
    "sunlight_exposure": "Exposición solar", "meals_per_day": "Comidas por día", "caffeine_intake": "Ingesta de cafeína",
    "family_history": "Antecedentes familiares", "pet_owner": "Posee mascota", "electrolyte_level": "Nivel de electrolitos",
    "gene_marker_flag": "Marcador genético", "environmental_risk_score": "Riesgo ambiental",
    "daily_supplement_dosage": "Dosis diaria de suplementos",
}
def pretty(col: str) -> str:
    if col in CUSTOM_ALIASES:
        return CUSTOM_ALIASES[col]
    nice = col.replace("_", " ").strip().capitalize()
    return nice.replace("Imc", "IMC")

# Columnas a considerar
cols = data.columns.drop('target', errors='ignore')
num_cols = data[cols].select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in cols if c not in num_cols]

# =========================
# Numéricas vs categóricas
# =========================
st.subheader("Distribución de variables por tipo")
counts_types = {"Numéricas": len(num_cols), "Categóricas": len(cat_cols)}
if counts_types["Numéricas"] + counts_types["Categóricas"] == 0:
    st.warning("No hay columnas para analizar (o todas fueron excluidas).")
else:
    cmap = plt.get_cmap("viridis")
    pie_colors = cmap(np.linspace(0.2, 0.8, 2))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)  # tamaño fijo
    ax.pie(
        [counts_types["Numéricas"], counts_types["Categóricas"]],
        labels=["Numéricas", "Categóricas"],
        autopct="%1.1f%%",
        startangle=90,
        colors=pie_colors,
        wedgeprops=dict(edgecolor="white", linewidth=1)
    )
    ax.axis("equal")
    ax.set_title("Variables numéricas vs categóricas (excluye `target`)")

    fig.tight_layout()

    # Centrar la figura en la página usando columnas
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.pyplot(fig, clear_figure=True, use_container_width=False)

st.markdown("""
Se puede observar la distribución de las variables del dataset por tipo. Donde predominan las variables numéricas (60,5%) frente a las categóricas (39,5%). Esta proporción sugiere que la mayor parte de la información disponible proviene de medidas continuas o discretas cuantificables (p. ej., edad, frecuencia cardiaca, pasos diarios), mientras que una porción relevante corresponde a atributos cualitativos (p. ej., género, tipo de ejercicio, nivel educativo).

Desde el punto de vista analítico, este balance tiene implicaciones directas en el preprocesamiento y el modelado. Las variables numéricas van a pasar por técnicas de imputación por mediana y escalado , además de ser buenas candidatas para reducción de dimensionalidad (PCA). Las variables categóricas, por su parte, exigirán codificación (one-hot) y un control del crecimiento de dimensiones para evitar la explosión de columnas. En conjunto.
""")

with st.expander("Mapeo de Variables"):
    mapping_df = pd.DataFrame({
        "columna_original": list(cols),
        "nombre_mostrado": [pretty(c) for c in cols]
    })
    st.dataframe(mapping_df, use_container_width=True)
    st.download_button(
        "Descargar CSV de mapeo",
        data=mapping_df.to_csv(index=False).encode("utf-8"),
        file_name="mapeo_nombres_columnas.csv",
        mime="text/csv"
    )


# =========================
# Objetivo y alcance
# =========================
st.markdown("""
## Objetivo del Proyecto y posibles usos del set de datos
- Descripción de set de datos
- Exploración de los datos
- Preprocesamiento (imputación, codificación).
- Selección de características.
- Disminución de dimensionalidad.
- Selección de los Modelos. 


## Alcance y Limitaciones
Proyecto con fines educativos y de investigación; **no** es una herramienta de diagnóstico que se pueda utilizar para predecir información.
""")

st.markdown("## Ver vista previa de los datos")

st.markdown("""
Los datos a trabajar son los siguientes, no se tienen en cuenta las siguientes variables:
 - survey_code: es el identificador de la encuesta
 - bmi_estimated: Ya con el BMI es suficiente y no trae información relevante para el modelo
 - bmi_scaled: Ya con el BMI es suficiente y no trae información relevante para el modelo
 - bmi_corrected: Ya con el BMI es suficiente y no trae información relevante para el modelo
""")

with st.expander("Ver vista previa de los datos"):
    st.dataframe(data.head(50), use_container_width=True)

# =========================
# Exploración visual de datos
# =========================
st.markdown("## Exploración visual de datos")

st.markdown("""
Antes de inicar con el proceso de visualización de datos, es importante tener en cuenta que los siguientes datos descriptivos de las variables:
""")

# =========================
# Estadísticos descriptivos 
# =========================
with st.expander("Estadísticos descriptivos (todas las columnas)", expanded=False):
    desc = data.describe(include="all").T
    
    # Reemplazar los nombres
    desc.index = [CUSTOM_ALIASES.get(col, col) for col in desc.index]

    st.dataframe(desc, use_container_width=True)


# =========================
# Utilidades
# =========================
def _viridis(n: int):
    cmap = plt.get_cmap("viridis")
    return cmap(np.linspace(0.2, 0.9, max(n, 1)))

def _pos_label_from_target(s: pd.Series) -> str:
    s = s.dropna().astype(str)
    if s.empty:
        return ""
    vc = s.value_counts()
    return "diseased" if "diseased" in vc.index else vc.index[-1]

def _risk_by_quantiles(df: pd.DataFrame, xcol: str, ycol: str = "target", q: int = 10):
    x = pd.to_numeric(df[xcol], errors="coerce")
    y = df[ycol].astype(str)
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if x.nunique() < 2:
        return pd.DataFrame(columns=["center","rate","n"])
    pos = _pos_label_from_target(y)
    bins = pd.qcut(x, q=min(q, x.nunique()), duplicates="drop")
    g = pd.DataFrame({"bin": bins, "is_pos": (y == pos).values})
    out = g.groupby("bin", observed=False).agg(rate=("is_pos","mean"), n=("is_pos","size")).reset_index(drop=True)
    centers = x.groupby(bins).median().values
    out["center"] = centers
    return out[["center","rate","n"]]

# =========================
# Grilla
# =========================
fig, axs = plt.subplots(2, 2, figsize=(18, 14), dpi=120)
axs = axs.flatten()

has_target = "target" in data.columns and data["target"].notna().any()
n_target = data["target"].nunique() if has_target else 0
if has_target:
    target_labels = sorted(data["target"].dropna().astype(str).unique().tolist())
    tgt_colors = _viridis(len(target_labels))

# ---- (1) Histograma: BMI (hue = target si hay ≤10 clases)
ax = axs[0]
if "bmi" in data.columns:
    bmi_vals = pd.to_numeric(data["bmi"], errors="coerce")
    if has_target and n_target <= 10:
        for lab, col in zip(target_labels, tgt_colors):
            vals = bmi_vals[data["target"].astype(str) == lab].dropna()
            if len(vals) > 0:
                ax.hist(vals, bins=30, alpha=0.55, color=col,
                        label=str(lab), edgecolor="white", linewidth=0.4)
        ax.legend(title="target", frameon=False)
    else:
        ax.hist(bmi_vals.dropna(), bins=30, color=_viridis(1)[0],
                edgecolor="white", linewidth=0.4)
    ax.set_title("Distribución de IMC")
    ax.set_xlabel(pretty("bmi")); ax.set_ylabel("Frecuencia")
    ax.grid(True, linewidth=0.3, alpha=0.4)
else:
    ax.text(0.5, 0.5, "Falta columna 'bmi'", ha="center", va="center")

# ---- (2) Dispersión: Edad vs Peso (hue = target si hay ≤10 clases)
ax = axs[1]
need = [c for c in ["age", "weight"] if c in data.columns]
if len(need) == 2:
    df_sc = data[need + (["target"] if has_target else [])].copy()
    df_sc["age"] = pd.to_numeric(df_sc["age"], errors="coerce")
    df_sc["weight"] = pd.to_numeric(df_sc["weight"], errors="coerce")
    df_sc = df_sc.dropna(subset=["age", "weight"])
    if has_target and n_target <= 10:
        for lab, col in zip(target_labels, tgt_colors):
            sub = df_sc[df_sc["target"].astype(str) == lab]
            ax.scatter(sub["age"], sub["weight"], s=12, alpha=0.6, c=[col], label=str(lab))
        ax.legend(title="target", frameon=False)
    else:
        ax.scatter(df_sc["age"], df_sc["weight"], s=12, alpha=0.6, c=[_viridis(1)[0]])
    ax.set_title("Dispersión: Edad vs Peso")
    ax.set_xlabel(pretty("age")); ax.set_ylabel(pretty("weight"))
    ax.grid(True, linewidth=0.3, alpha=0.4)
else:
    ax.text(0.5, 0.5, "Faltan 'age' o 'weight'", ha="center", va="center")

# ---- (3) Curvas de riesgo por cuantiles (Age, BMI, Glucose, Sleep hours)
ax = axs[2]
vars_plot = [("age","Edad"), ("bmi","IMC"), ("glucose","Glucosa"), ("sleep_hours","Horas de sueño")]
vars_plot = [v for v in vars_plot if v[0] in data.columns]
colors_lines = _viridis(len(vars_plot))
if has_target and vars_plot:
    for (col, _), col_color in zip(vars_plot, colors_lines):
        df_q = _risk_by_quantiles(data, col, "target", q=10)
        if not df_q.empty:
            ax.plot(df_q["center"], df_q["rate"], marker="o", linestyle="--",
                    linewidth=1.2, color=col_color, label=pretty(col))
    ax.set_ylim(0, 1)
    ax.set_title("Tasa de 'diseased' por cuantiles")
    ax.set_xlabel("Centro del cuantil"); ax.set_ylabel("Proporción 'diseased'")
    ax.legend(frameon=False)
    ax.grid(True, linewidth=0.3, alpha=0.4)
else:
    ax.text(0.5, 0.5, "Falta 'target' o variables numéricas", ha="center", va="center")

# ---- (4) Dispersión: Pasos diarios vs Actividad física (hue = target si hay ≤10 clases)
ax = axs[3]
cols_needed = [c for c in ["daily_steps", "physical_activity", "target"] if c in data.columns]
if all(c in data.columns for c in ["daily_steps", "physical_activity"]):
    df_sc2 = data[cols_needed].copy()
    df_sc2["daily_steps"] = pd.to_numeric(df_sc2["daily_steps"], errors="coerce")
    df_sc2["physical_activity"] = pd.to_numeric(df_sc2["physical_activity"], errors="coerce")
    df_sc2 = df_sc2.dropna(subset=["daily_steps", "physical_activity"])
    if has_target and n_target <= 10 and "target" in df_sc2.columns:
        labels = sorted(df_sc2["target"].dropna().astype(str).unique().tolist())
        cols = plt.get_cmap("viridis")(np.linspace(0.2, 0.9, len(labels)))
        for lab, col in zip(labels, cols):
            sub = df_sc2[df_sc2["target"].astype(str) == lab]
            ax.scatter(sub["daily_steps"], sub["physical_activity"], s=12, alpha=0.6, c=[col], label=str(lab))
        ax.legend(title="target", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        ax.scatter(df_sc2["daily_steps"], df_sc2["physical_activity"], s=12, alpha=0.6, c=[plt.get_cmap("viridis")(0.6)])
    ax.set_xlabel(pretty("daily_steps"))
    ax.set_ylabel(pretty("physical_activity"))
    ax.set_title("Dispersión: Pasos diarios vs Actividad física")
    ax.grid(True, linewidth=0.3, alpha=0.4)
else:
    ax.text(0.5, 0.5, "Faltan 'daily_steps' o 'physical_activity'", ha="center", va="center")

# Ajustar y centrar en la página
fig.tight_layout(pad=3.0)
_, col_mid, _ = st.columns([1, 3, 1])
with col_mid:
    st.pyplot(fig, clear_figure=True, use_container_width=False)

st.markdown("""
En el histograma de IMC, ambas clases muestran una distribución aproximadamente normal con moda en el rango 22–26. Se observa un ligero desplazamiento de la clase diseased hacia valores de IMC más altos y colas más largas por encima de 30, aunque el solapamiento es elevado; por sí solo, el IMC no separa completamente las clases.

La dispersión Edad vs. Peso evidencia una relación débil entre ambas variables: el peso se concentra entre 60 y 110 kg a lo largo de todas las edades, con mezcla de clases en casi todo el rango. Esto sugiere que métricas normalizadas por talla (p. ej., IMC).

El gráfico de tasa de “diseased” por cuantiles (edad, IMC, glucosa y horas de sueño) muestra líneas casi planas alrededor de 0,3. Se ve un leve aumento de la tasa en cuantiles altos de IMC y glucosa, mientras que edad y sueño presentan variaciones mínimas. La señal univariada es moderada o no lineal.

Por último, en pasos diarios vs. actividad física aparece una tendencia positiva (más pasos, mayor nivel de actividad). Se percibe mayor densidad de healthy en rangos medios–altos de pasos (>7.000 aprox.) y actividad, aunque persiste un solapamiento considerable con diseased en niveles bajos y medios.
            
            """)


# =========================
# Procesamiento de datos
# =========================

st.markdown("## Procesamiento de datos")

# Definir X (características) y y (objetivo) de forma segura
drop_cols = ['survey_code', 'bmi_estimated', 'bmi_scaled', 'bmi_corrected', 'target']
X = data.drop(columns=[c for c in drop_cols if c in data.columns], errors='ignore')
y = data['target'] if 'target' in data.columns else None

# Dividir el grupo de testeo y entrenamiento con el 80%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.80, stratify=y, random_state=42
)

st.subheader("Distribución de las variables")

def dist_table(s: pd.Series) -> pd.DataFrame:
    """Retorna conteo y proporción por clase, ordenado por etiqueta."""
    counts = s.value_counts(dropna=False)
    props  = s.value_counts(normalize=True, dropna=False)
    out = pd.DataFrame({"count": counts, "proportion": props.round(3)})
    return out.sort_index()

tab_total, tab_train, tab_test = st.tabs(["Total", "Train", "Test"])

with tab_total:
    st.caption("Proporción total")
    st.dataframe(dist_table(y), use_container_width=True)

with tab_train:
    st.caption("Proporción en el conjunto de entrenamiento")
    st.dataframe(dist_table(y_train), use_container_width=True)

with tab_test:
    st.caption("Proporción en el conjunto de prueba")
    st.dataframe(dist_table(y_test), use_container_width=True)


# Detectar tipos de columnas
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

# Codificar y (sin imprimir)
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train) if y_train is not None else None
y_test_enc  = label_encoder.transform(y_test) if y_test is not None else None

# Pipelines de preprocesamiento
num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
    ],
    sparse_threshold=0.0,
    remainder="drop"
)

# Pipeline final (listo para encadenar modelo después)
model_pipeline = Pipeline(steps=[
    ("pre", preprocessor)
])

# =========================
# Diagrama del pipeline
# =========================
st.markdown("""
**Diagrama del pipeline de preprocesamiento**
""")

# Asegurar que existan num_cols y cat_cols (por si aún no están definidos)
if "num_cols" not in locals() or "cat_cols" not in locals():
    # Reconstruir X como en tu pipeline
    drop_cols = ['survey_code', 'bmi_estimated', 'bmi_scaled', 'bmi_corrected', 'target']
    X_tmp = data.drop(columns=[c for c in drop_cols if c in data.columns], errors='ignore')
    num_cols = X_tmp.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_tmp.select_dtypes(exclude=[np.number]).columns.tolist()

# Texto auxiliar (muestra hasta 6 columnas por tipo)
def preview_cols(cols, n=6):
    if not cols: 
        return "—"
    cols = [str(c) for c in cols[:n]]
    return ", ".join(cols) + ("…" if len(cols) > n else "")

num_count = len(num_cols)
cat_count = len(cat_cols)
num_preview = preview_cols(num_cols)
cat_preview = preview_cols(cat_cols)

n_filas = len(data)

dot = f'''
digraph G {{
  rankdir=LR;
  graph [pad="0.3", nodesep="0.4", ranksep="0.6"];
  node  [shape=box, style="rounded,filled", color="#4F46E5", fillcolor="#EEF2FF", fontname="Helvetica"];
  edge  [color="#64748B"];

  X [label="Datos (X)\\n{n_filas} filas"];
  PRE [label="Preprocesamiento (ColumnTransformer)"];

  X -> PRE;

  subgraph cluster_num {{
    style="rounded"; color="#CBD5E1"; label="Numéricas ({num_count})\\n{num_preview}";
    NUM_IMP   [label="SimpleImputer\\n(strategy=median)"];
    NUM_SCALE [label="StandardScaler"];
    NUM_IMP -> NUM_SCALE;
  }}

  subgraph cluster_cat {{
    style="rounded"; color="#CBD5E1"; label="Categóricas ({cat_count})\\n{cat_preview}";
    CAT_IMP [label="SimpleImputer\\n(strategy=most_frequent)"];
    CAT_OHE [label="OneHotEncoder\\n(handle_unknown=ignore)"];
    CAT_IMP -> CAT_OHE;
  }}

  PRE -> NUM_IMP;
  PRE -> CAT_IMP;

  OUT [label="Salida preprocesada\\n(Matriz densa)"];
  NUM_SCALE -> OUT;
  CAT_OHE   -> OUT;
}}
'''

st.graphviz_chart(dot, use_container_width=True)

st.markdown("""
Se observa que:

- Variables numéricas (edad, peso, IMC, etc.) son tratadas mediante SimpleImputer (con estrategia de mediana) para manejar valores faltantes, seguido de un StandardScaler para normalizar los datos.
- Variables categóricas (género, calidad de sueño, consumo de alcohol, etc.) también se imputan con SimpleImputer (estrategia de más frecuente) para completar los valores faltantes, seguidas de una codificación OneHotEncoder para convertir las categorías en variables binarias.
- El resultado final es una matriz densa de características, lista para ser utilizada en la modelización.
""")

# =========================
# Técnicas de Selección de Variables — Gráficas comparativas (Chi² + ANOVA + RF + RFECV)
# =========================
st.markdown("## Técnicas de Selección de Variables — Gráficas comparativas")

# Umbral global para selección
UMBRAL_SELECCION = 0.90
LABEL_UMBRAL = f"{int(UMBRAL_SELECCION*100)}% acumulado"

# ---------- Helpers ----------
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif  # ANOVA

def onehot_dense(**kw):
    """OneHotEncoder compatible con distintas versiones (sparse_output vs sparse)."""
    try:
        return OneHotEncoder(sparse_output=False, **kw)  # sklearn >= 1.2
    except TypeError:
        return OneHotEncoder(sparse=False, **kw)         # sklearn < 1.2

def get_feature_names(ct):
    """Obtiene nombres de features desde ColumnTransformer (compatibilidad amplia)."""
    try:
        return ct.get_feature_names_out()
    except Exception:
        names = []
        for name, trans, cols in ct.transformers_:
            if name == "remainder":
                continue
            if hasattr(trans, "named_steps"):
                if "ohe" in trans.named_steps:
                    ohe = trans.named_steps["ohe"]
                    try:
                        fn = ohe.get_feature_names_out(cols)
                    except Exception:
                        fn = ohe.get_feature_names(cols)
                    names.extend(fn)
                elif "kbins" in trans.named_steps:
                    # 3 bins por KBinsDiscretizer
                    names.extend([f"{c}_bin{i}" for c in cols for i in range(3)])
                else:
                    names.extend(list(cols))
            else:
                names.extend(list(cols))
        return np.array(names)

def make_train_split(df, seed=42):
    """Split sin muestreo: usa todo el dataset (80/20 estratificado)."""
    drop_cols = ['survey_code', 'bmi_estimated', 'bmi_scaled', 'bmi_corrected']
    X_ = df.drop(columns=[c for c in drop_cols + ['target'] if c in df.columns], errors='ignore')
    if 'target' not in df.columns:
        st.warning("No se encontró la columna `target` en los datos.")
        st.stop()
    y_ = df['target']
    return train_test_split(X_, y_, train_size=0.80, stratify=y_, random_state=seed)

def _safe_cumsum_cutoff(sorted_scores, umbral):
    denom = float(np.sum(sorted_scores))
    if denom <= 0 or len(sorted_scores) == 0:
        return np.array([]), 0
    cum = np.cumsum(sorted_scores) / denom
    cutoff = int(np.searchsorted(cum, umbral) + 1)
    return cum, cutoff


X_train, X_test, y_train, y_test = make_train_split(data)
num_cols_base = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_base = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
y_enc = LabelEncoder().fit_transform(y_train)

# ==========================================================
# 1) FILTRADO (Chi²)
# ==========================================================
with st.spinner("Calculando método de Filtrado (Chi²)..."):
    nuniq = X_train[num_cols_base].nunique()
    num_to_bin = nuniq[nuniq > 3].index.tolist()   # numéricas continuas -> KBins + OHE
    num_small  = nuniq[nuniq <= 3].index.tolist()  # numéricas discretas pequeñas -> OHE

    transformers_chi2 = []
    if num_to_bin:
        transformers_chi2.append((
            "num_bin",
            Pipeline(steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("kbins", KBinsDiscretizer(n_bins=3, strategy="quantile", encode="onehot"))
            ]),
            num_to_bin
        ))
    if num_small:
        transformers_chi2.append((
            "num_small",
            Pipeline(steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", onehot_dense(handle_unknown="ignore"))
            ]),
            num_small
        ))
    if cat_cols_base:
        transformers_chi2.append((
            "cat",
            Pipeline(steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", onehot_dense(handle_unknown="ignore"))
            ]),
            cat_cols_base
        ))

    preprocess_chi2 = ColumnTransformer(
        transformers=transformers_chi2,
        remainder="drop",
        sparse_threshold=0.3
    )
    X_for_chi2 = preprocess_chi2.fit_transform(X_train)
    feat_names_chi2 = get_feature_names(preprocess_chi2)

    sel_chi2 = SelectKBest(score_func=chi2, k='all').fit(X_for_chi2, y_enc)
    scores_chi2 = np.nan_to_num(sel_chi2.scores_, nan=0.0)
    idx_chi2 = np.argsort(scores_chi2)[::-1]
    sorted_scores_chi2 = scores_chi2[idx_chi2]
    sorted_features_chi2 = np.array(feat_names_chi2)[idx_chi2]
    cum_chi2, cutoff_chi2 = _safe_cumsum_cutoff(sorted_scores_chi2, UMBRAL_SELECCION)

# ==========================================================
# 1b) FILTRADO (ANOVA f_classif)
# ==========================================================
with st.spinner("Calculando método de Filtrado (ANOVA)..."):
    if len(num_cols_base) > 0:
        preprocess_anova = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]), num_cols_base),
            ],
            remainder="drop",
            sparse_threshold=0.3
        )
        X_for_anova = preprocess_anova.fit_transform(X_train)
        feat_names_anova = get_feature_names(preprocess_anova)

        sel_anova = SelectKBest(score_func=f_classif, k='all').fit(X_for_anova, y_enc)
        scores_anova = np.nan_to_num(sel_anova.scores_, nan=0.0)
        idx_anova = np.argsort(scores_anova)[::-1]
        sorted_scores_anova = scores_anova[idx_anova]
        sorted_features_anova = np.array(feat_names_anova)[idx_anova]
        cum_anova, cutoff_anova = _safe_cumsum_cutoff(sorted_scores_anova, UMBRAL_SELECCION)
    else:
        sorted_scores_anova = np.array([])
        sorted_features_anova = np.array([])
        cum_anova = np.array([])
        cutoff_anova = 0

# ==========================================================
# 2) INCRUSTADA (RandomForest feature_importances_)
# ==========================================================
with st.spinner("Calculando método Incrustado (RandomForest)..."):
    pre_rf = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols_base),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("ohe", onehot_dense(handle_unknown="ignore"))]), cat_cols_base),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )
    X_enc_rf = pre_rf.fit_transform(X_train)
    feat_names_rf = get_feature_names(pre_rf)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_enc_rf, y_train)
    importances_emb = rf.feature_importances_
    idx_emb = np.argsort(importances_emb)[::-1]
    sorted_importances_emb = importances_emb[idx_emb]
    sorted_features_emb = np.array(feat_names_rf)[idx_emb]
    cum_emb, cutoff_emb = _safe_cumsum_cutoff(sorted_importances_emb, UMBRAL_SELECCION)

# ==========================================================
# 3) ENVOLTURA (RFECV con Regresión Logística)
# ==========================================================
with st.spinner("Calculando método por Envoltura (RFECV)..."):
    num_pipeline_rfe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline_rfe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", onehot_dense(handle_unknown="ignore"))
    ])
    preprocessor_rfe = ColumnTransformer(
        transformers=[
            ("num", num_pipeline_rfe, num_cols_base),
            ("cat", cat_pipeline_rfe, cat_cols_base),
        ],
        remainder="drop"
    )
    X_pre = preprocessor_rfe.fit_transform(X_train)
    p = X_pre.shape[1]
    min_feats = max(5, int(np.sqrt(max(p, 1))))

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    est = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1500)

    rfecv = RFECV(
        estimator=est,
        step=1,
        cv=cv,
        scoring="roc_auc",
        min_features_to_select=min_feats,
        n_jobs=-1
    )
    pipeline_rfe = Pipeline([
        ('preprocessor', preprocessor_rfe),
        ('feature_selection', rfecv)
    ])
    pipeline_rfe.fit(X_train, y_train)

    rfecv_res = pipeline_rfe.named_steps['feature_selection']
    feat_names_rfe = pipeline_rfe.named_steps['preprocessor'].get_feature_names_out()
    mask = rfecv_res.support_
    selected_features_rfe = feat_names_rfe[mask]

    coefs = rfecv_res.estimator_.coef_.ravel() if hasattr(rfecv_res.estimator_, "coef_") else np.zeros(len(selected_features_rfe))
    idx_wrap = np.argsort(np.abs(coefs))[::-1]
    abs_coefs_sorted = np.abs(coefs)[idx_wrap]
    features_wrap_sorted = selected_features_rfe[idx_wrap]
    cum_wrap, cutoff_wrap = _safe_cumsum_cutoff(abs_coefs_sorted, UMBRAL_SELECCION)

# =========================
# GRÁFICAS COMPARATIVAS
# =========================
with st.expander("Ver gráficas comparativas (Chi² / ANOVA / RF / RFECV)", expanded=False):
    TOP_K = 25
    cmap = plt.get_cmap("viridis")
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), dpi=120)
    axs = axs.flatten()

    # --- (1) Chi² ---
    k1 = min(TOP_K, len(sorted_scores_chi2))
    if k1 > 0:
        colors1 = cmap(np.linspace(0.2, 0.9, k1))
        axs[0].bar(range(k1), sorted_scores_chi2[:k1], color=colors1)
        axs[0].axvline(min(cutoff_chi2-1, max(k1-1, 0)), color='red', linestyle='--', label=LABEL_UMBRAL)
        axs[0].legend(frameon=False)
        axs[0].set_xticks(range(k1))
        axs[0].set_xticklabels(sorted_features_chi2[:k1], rotation=90, fontsize=8)
    else:
        axs[0].text(0.5, 0.5, "Sin datos", ha="center", va="center")
    axs[0].set_title("Filtrado (Chi²)")
    axs[0].set_ylabel("Puntuación")

    # --- (2) ANOVA ---
    kA = min(TOP_K, len(sorted_scores_anova))
    if kA > 0:
        colorsA = cmap(np.linspace(0.2, 0.9, kA))
        axs[1].bar(range(kA), sorted_scores_anova[:kA], color=colorsA)
        axs[1].axvline(min(cutoff_anova-1, max(kA-1, 0)), color='red', linestyle='--', label=LABEL_UMBRAL)
        axs[1].legend(frameon=False)
        axs[1].set_xticks(range(kA))
        axs[1].set_xticklabels(sorted_features_anova[:kA], rotation=90, fontsize=8)
    else:
        axs[1].text(0.5, 0.5, "Sin datos", ha="center", va="center")
    axs[1].set_title("Filtrado (ANOVA f_classif)")
    axs[1].set_ylabel("Estadístico F")

    # --- (3) RandomForest ---
    k2 = min(TOP_K, len(sorted_importances_emb))
    if k2 > 0:
        colors2 = cmap(np.linspace(0.2, 0.9, k2))
        axs[2].bar(range(k2), sorted_importances_emb[:k2], color=colors2)
        axs[2].axvline(min(cutoff_emb-1, max(k2-1, 0)), color='red', linestyle='--', label=LABEL_UMBRAL)
        axs[2].legend(frameon=False)
        axs[2].set_xticks(range(k2))
        axs[2].set_xticklabels(sorted_features_emb[:k2], rotation=90, fontsize=8)
    else:
        axs[2].text(0.5, 0.5, "Sin datos", ha="center", va="center")
    axs[2].set_title("Incrustada (RandomForest)")
    axs[2].set_ylabel("Importancia")

    # --- (4) RFECV ---
    k3 = min(TOP_K, len(abs_coefs_sorted))
    if k3 > 0:
        colors3 = cmap(np.linspace(0.2, 0.9, k3))
        axs[3].bar(range(k3), abs_coefs_sorted[:k3], color=colors3)
        axs[3].axvline(min(cutoff_wrap-1, max(k3-1, 0)), color='red', linestyle='--', label=LABEL_UMBRAL)
        axs[3].legend(frameon=False)
        axs[3].set_xticks(range(k3))
        labels3 = features_wrap_sorted[:k3] if len(features_wrap_sorted) >= k3 else features_wrap_sorted
        axs[3].set_xticklabels(labels3, rotation=90, fontsize=8)
    else:
        axs[3].text(0.5, 0.5, "Sin datos", ha="center", va="center")
    axs[3].set_title("Envoltura (RFECV)")
    axs[3].set_ylabel("|Coeficiente|")

    fig.tight_layout(pad=2.0)
    st.pyplot(fig, clear_figure=True, use_container_width=False)

# =========================
# Resumen de variables seleccionadas
# =========================
st.markdown("### Resumen de variables seleccionadas")
def safe_n_selected(cutoff, total):
    if total <= 0 or cutoff is None:
        return 0
    return int(min(max(cutoff, 0), total))

n_total_chi2 = len(sorted_features_chi2)
n_sel_chi2   = safe_n_selected(cutoff_chi2, n_total_chi2)

n_total_anova = len(sorted_features_anova)
n_sel_anova   = safe_n_selected(cutoff_anova, n_total_anova)

n_total_emb = len(sorted_features_emb)
n_sel_emb   = safe_n_selected(cutoff_emb, n_total_emb)

n_total_wrap = len(features_wrap_sorted) if 'features_wrap_sorted' in locals() else 0
n_sel_wrap   = safe_n_selected(cutoff_wrap, n_total_wrap)

col_sel = f"Variables seleccionadas ({int(UMBRAL_SELECCION*100)}%)"
resumen_df = pd.DataFrame({
    "Técnica": [
        "Filtrado (Chi²)",
        "Filtrado (ANOVA f_classif)",
        "Incrustada (RandomForest)",
        "Envoltura (RFECV)"
    ],
    col_sel: [n_sel_chi2, n_sel_anova, n_sel_emb, n_sel_wrap],
    "Total evaluadas": [n_total_chi2, n_total_anova, n_total_emb, n_total_wrap],
    "% del total": [
        f"{(n_sel_chi2 / n_total_chi2 * 100):.1f}%" if n_total_chi2 else "—",
        f"{(n_sel_anova / n_total_anova * 100):.1f}%" if n_total_anova else "—",
        f"{(n_sel_emb   / n_total_emb   * 100):.1f}%" if n_total_emb   else "—",
        f"{(n_sel_wrap  / n_total_wrap  * 100):.1f}%" if n_total_wrap  else "—",
    ],
})
st.dataframe(resumen_df, use_container_width=True)

st.markdown("""

En las cuatro técnicas (Chi², ANOVA, RandomForest y RFECV) se repiten variables algunas variables que explican el 90% de la varianza como lo es: IMC, peso, cintura, actividad/hábitos, pasos diarios, nivel/horas de trabajo o actividad, tipo de trabajo, calidad de sueño. Lo que se puede concluir que si se van a utilizar alguna de estas técnicas de selección. Se tienene los siguientes resultados:

Diferencias por enfoque:

- Filtro (Chi² / ANOVA): prioriza muchas variables con relación univariada; con tu umbral del 90% acumulado selecciona 53 (Chi²) y 8 (ANOVA) de las evaluadas. Útil para un primer cribado, pero puede retener variables redundantes.
- Incrustada (RandomForest): concentra la importancia en menos variables pero con relaciones no lineales e interacciones; con 90% acumulada. Quedan 52 de 79.
- Envoltura (RFECV): es la más restrictiva y termina con 10 de 12, lo que sugiere que un subconjunto pequeño ya explica gran parte del rendimiento del estimador usado.

Teniendo en cuenta estas 3 técnicas de filtrado, incrustación y envoltura se identifica una distribución entre varias variables correlacionadas; los métodos que consideran interacciones (RF/RFECV) confirman que no todo es lineal/univariado.

Por ello es se decide tomar la decisión de utilizar técnicas como lo son PCA y MCA
""")

# =========================
# Análisis de Componentes Principales (PCA)
# =========================
st.markdown("## Análisis de Componentes Principales (PCA)")

# --- Parámetro desde la UI ---
with st.sidebar:
    st.subheader("Parámetros PCA")
    umbral_pca_pct = st.slider("Umbral de varianza (%)", min_value=70, max_value=99, value=90, step=5, key="umbral_pca")
UMBRAL_VAR = umbral_pca_pct / 100.0
LABEL_VAR  = f"{umbral_pca_pct}%"

# --- Seleccionar solo variables numéricas ---
num_cols_pca = data.select_dtypes(include=np.number).columns.tolist()
if len(num_cols_pca) < 2:
    st.warning("No hay suficientes variables numéricas para realizar PCA (se requieren al menos 2).")
    st.stop()

# --- Imputación y escalado ---
num_imputer = SimpleImputer(strategy="median")
scaler      = StandardScaler()

X_num      = num_imputer.fit_transform(data[num_cols_pca])
X_num_std  = scaler.fit_transform(X_num)

# --- PCA completo ---
pca = PCA()
X_pca = pca.fit_transform(X_num_std)

# --- Varianza explicada ---
explained_var = pca.explained_variance_ratio_            # proporciones [0..1]
explained_pct = explained_var * 100.0                    # en %
cum_var = np.cumsum(explained_var)

# --- Nº de componentes para cubrir el umbral ---
n_components = int(np.searchsorted(cum_var, UMBRAL_VAR) + 1)

# Paleta viridis
cmap = plt.get_cmap("viridis")

# =========================
# Fila 1: Grilla 1×2 (varianza acumulada | PC1 vs PC2)
# =========================
col_left, col_right = st.columns(2)

# --- (1) Varianza acumulada ---
with col_left:
    fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=120)
    ax1.plot(range(1, len(cum_var) + 1), cum_var, marker="o", linestyle="--",
             color=cmap(0.7), label="Varianza acumulada")
    ax1.axhline(y=UMBRAL_VAR, color="r", linestyle="--", label=f"Umbral {LABEL_VAR}")
    ax1.axvline(x=n_components, color="g", linestyle="--", label=f"{n_components} componentes")
    ax1.set_xlabel("Número de Componentes Principales")
    ax1.set_ylabel("Varianza explicada acumulada")
    ax1.set_title("PCA en variables numéricas")
    ax1.set_ylim(0, 1)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", frameon=False)
    fig1.tight_layout()
    st.pyplot(fig1, clear_figure=True, use_container_width=False)

# --- (2) PC1 vs PC2 (con % en ejes y var. acumulada anotada) ---
with col_right:
    fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=120)

    # Etiquetas de ejes con % de varianza
    pc1_label = f"PC1 ({explained_pct[0]:.1f}%)" if len(explained_pct) > 0 else "PC1"
    pc2_label = f"PC2 ({explained_pct[1]:.1f}%)" if len(explained_pct) > 1 else "PC2"
    cum2 = (explained_var[:2].sum() * 100.0) if len(explained_var) >= 2 else (explained_var[:1].sum() * 100.0)

    if "target" in data.columns:
        labels = data["target"].astype(str).fillna("NA")
        uniq = sorted(labels.unique())
        colors = dict(zip(uniq, cmap(np.linspace(0.2, 0.9, len(uniq)))))
        for cls in uniq:
            mask = (labels == cls).to_numpy()
            ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.6, edgecolor="k",
                        label=str(cls), color=colors[cls], s=14)
        ax2.legend(title="Estado de salud", loc="best", frameon=False)
    else:
        ax2.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, edgecolor="k", color=cmap(0.6), s=14)

    # Ejes + título
    ax2.set_xlabel(pc1_label)
    ax2.set_ylabel(pc2_label)
    ax2.set_title("Dispersión en el espacio de los dos primeros componentes")
    ax2.grid(True, linestyle="--", alpha=0.4)

    # Anotar varianza acumulada PC1+PC2 dentro del gráfico
    ax2.text(
        0.02, 0.98,
        f"Var. acumulada PC1+PC2 = {cum2:.1f}%",
        transform=ax2.transAxes, va="top", ha="left",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7)
    )

    fig2.tight_layout()
    st.pyplot(fig2, clear_figure=True, use_container_width=False)

# =========================
# Fila 2 (centrada): Heatmap de loadings AGRANDADO con valores
# =========================
st.subheader(f"Heatmap de loadings (primeros {n_components} componentes - {LABEL_VAR} varianza)")

# Re-entrenar PCA con n_components óptimos (para loadings compactos)
pca_opt = PCA(n_components=n_components)
X_pca_opt = pca_opt.fit_transform(X_num_std)

loadings = pd.DataFrame(
    pca_opt.components_.T,
    index=num_cols_pca,
    columns=[f"PC{i+1}" for i in range(n_components)]
)

# Tamaño dinámico AGRANDADO según #variables y #componentes
n_vars, n_comps = loadings.shape
fig3_width  = min(22, max(10, 0.8 * n_comps + 10))     # ancho crece con #PCs
fig3_height = min(18, max(8, 0.28 * n_vars + 6))       # alto crece con #vars

fig3, ax3 = plt.subplots(figsize=(fig3_width, fig3_height), dpi=120)

# Heatmap con anotaciones de valores
sns.heatmap(
    loadings,
    cmap="viridis", center=0,
    annot=True, fmt=".2f",  # ← valores en cada celda
    annot_kws={"size": 8},
    cbar_kws={"label": "Loading"}
)
ax3.set_xlabel("Componentes Principales")
ax3.set_ylabel("Variables numéricas")
ax3.set_title(f"Loadings de PCA (hasta {LABEL_VAR} de varianza)")

# Mejorar legibilidad de etiquetas
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right")
ax3.set_yticklabels(ax3.get_yticklabels(), rotation=0)
fig3.tight_layout()

# Centrar el heatmap abajo
_, col_mid, _ = st.columns([1, 5, 1])
with col_mid:
    st.pyplot(fig3, clear_figure=True, use_container_width=False)

st.markdown("""
- *PCA – Varianza acumulada variables numericas:* La curva muestra que con 20 componentes se supera el 90% de varianza explicada (línea roja). Implica una reducción sustancial de dimensionalidad manteniendo la mayor parte de la información, ya que se reduce de 26 variables numericas para explicar el 90% de la varianza.

- *Dispersión PC1 vs PC2:* Las dos primeras componentes generan una nube muy solapada entre healthy y diseased, donde se representa un 13.1% entre las dos. por sí solas no separan los datos en la variables objetivo.

- *Heatmap de loadings (20 PCs):* Los pesos de las variables sobre cada componente están repartidos, sin un único dominador. Se distinguen bloques como: medidas corporales (p. ej., weight/ BMI/ waist_size) cargan fuerte en algunas PCs; hábitos/actividad (daily_steps, physical_activity) y biomarcadores (glucose, blood_pressure, cholesterol) contribuyen en otras. Esto confirma correlaciones entre grupos y justifica el uso de PCA para compactarlas.
""")

# =========================
# Análisis de Correspondencias Múltiples (MCA)
# =========================
st.markdown("## Análisis de Correspondencias Múltiples (MCA)")

# --- Parámetro desde la UI (clave única para evitar colisiones) ---
with st.sidebar:
    st.subheader("Parámetros MCA")
    umbral_mca_pct = st.slider(
        "Umbral de varianza MCA (%)", min_value=70, max_value=99, value=90, step=5, key="umbral_mca_sel"
    )
UMBRAL_MCA = umbral_mca_pct / 100.0
LABEL_MCA  = f"{umbral_mca_pct}%"

# Paleta viridis
cmap = plt.get_cmap("viridis")

# --- Utilidades compatibles con múltiples versiones de prince ---
def get_inertia_array(model):
    """Proporción de inercia explicada por dimensión (vector)."""
    if hasattr(model, "explained_inertia_") and model.explained_inertia_ is not None:
        arr = np.asarray(model.explained_inertia_, dtype=float)
        if arr.size:
            return arr
    if hasattr(model, "eigenvalues_") and model.eigenvalues_ is not None:
        ev = np.asarray(model.eigenvalues_, dtype=float); s = ev.sum()
        return ev / s if s > 0 else np.zeros_like(ev)
    if hasattr(model, "singular_values_") and model.singular_values_ is not None:
        sv = np.asarray(model.singular_values_, dtype=float); ev = sv**2; s = ev.sum()
        return ev / s if s > 0 else np.zeros_like(ev)
    return np.array([])

def get_row_coords(model, df_cat):
    if hasattr(model, "row_coordinates"):
        return model.row_coordinates(df_cat).to_numpy()
    if hasattr(model, "row_principal_coordinates"):
        return model.row_principal_coordinates(df_cat).to_numpy()
    return None

def get_col_coords(model, df_cat):
    if hasattr(model, "column_coordinates"):
        return model.column_coordinates(df_cat)
    if hasattr(model, "column_principal_coordinates"):
        return model.column_principal_coordinates(df_cat)
    return None

# --- Tomar solo columnas categóricas originales ---
cat_cols_mca = data.select_dtypes(exclude=[np.number]).columns.tolist()
if len(cat_cols_mca) == 0:
    st.info("No se detectaron variables categóricas para MCA.")
    st.stop()

# --- Imputación simple (moda) y asegurar tipo string ---
cat_imputer = SimpleImputer(strategy="most_frequent")
data_cat = pd.DataFrame(
    cat_imputer.fit_transform(data[cat_cols_mca]),
    columns=cat_cols_mca,
    index=data.index
).astype(str)

# --- Ajustar MCA con prince ---
with st.spinner("Calculando MCA..."):
    mca_pr = prince.MCA(
        n_components=min(50, len(cat_cols_mca) * 3),
        n_iter=5,
        copy=True,
        check_input=True,
        random_state=42
    ).fit(data_cat)

    inertia = get_inertia_array(mca_pr)  # proporción por dimensión (0..1)
    if inertia.size == 0:
        st.warning("No fue posible obtener la inercia explicada del MCA (revisa la versión de `prince`).")
        st.stop()

    inertia_pct = inertia * 100.0
    cum_inertia = np.cumsum(inertia)
    n_components_mca = int(np.searchsorted(cum_inertia, UMBRAL_MCA) + 1)

# =========================
# Fila 1: Grilla
# =========================
col_left, col_right = st.columns(2)

# --- (1) Varianza acumulada ---
with col_left:
    fig_mca1, ax_mca1 = plt.subplots(figsize=(8, 5), dpi=120)
    ax_mca1.plot(range(1, len(cum_inertia) + 1), cum_inertia, marker="o", linestyle="--", color=cmap(0.7),
                 label="Varianza acumulada")
    ax_mca1.axhline(y=UMBRAL_MCA, color="r", linestyle="--", label=f"Umbral {LABEL_MCA}")
    ax_mca1.axvline(x=n_components_mca, color="g", linestyle="--", label=f"{n_components_mca} dim.")
    ax_mca1.set_xlabel("Dimensiones MCA")
    ax_mca1.set_ylabel("Varianza explicada acumulada")
    ax_mca1.set_title("Varianza acumulada MCA (categóricas)")
    ax_mca1.set_ylim(0, 1)
    ax_mca1.grid(True, linestyle="--", alpha=0.5)
    ax_mca1.legend(loc="lower right", frameon=False)
    fig_mca1.tight_layout()
    st.pyplot(fig_mca1, clear_figure=True, use_container_width=False)

# --- (2) Dispersión Dim 1 vs Dim 2 con % en ejes y var. acumulada anotada ---
with col_right:
    row_coords = get_row_coords(mca_pr, data_cat)
    fig_mca2, ax_mca2 = plt.subplots(figsize=(8, 5), dpi=120)
    if row_coords is None or row_coords.shape[1] < 2:
        ax_mca2.text(0.5, 0.5, "No se pudieron obtener coordenadas de filas para el scatter.",
                     ha="center", va="center")
    else:
        dim1_pct = inertia_pct[0] if len(inertia_pct) > 0 else 0.0
        dim2_pct = inertia_pct[1] if len(inertia_pct) > 1 else 0.0
        cum2_pct = dim1_pct + dim2_pct

        if "target" in data.columns:
            labels = data.loc[data_cat.index, "target"].astype(str).fillna("NA")
            uniq = sorted(labels.unique())
            colors = dict(zip(uniq, cmap(np.linspace(0.2, 0.9, len(uniq)))))
            for cls in uniq:
                msk = (labels == cls).to_numpy()
                ax_mca2.scatter(row_coords[msk, 0], row_coords[msk, 1],
                                alpha=0.7, edgecolor="k", label=str(cls), color=colors[cls], s=14)
            ax_mca2.legend(title="Clase", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
        else:
            ax_mca2.scatter(row_coords[:, 0], row_coords[:, 1], alpha=0.7, edgecolor="k", color=cmap(0.6), s=14)

        ax_mca2.set_xlabel(f"Dimensión 1 ({dim1_pct:.1f}%)")
        ax_mca2.set_ylabel(f"Dimensión 2 ({dim2_pct:.1f}%)")
        ax_mca2.set_title("Scatter MCA — Dim 1 vs Dim 2")
        ax_mca2.grid(True, linestyle="--", alpha=0.4)

        # Varianza acumulada Dim1+Dim2
        ax_mca2.text(
            0.02, 0.98,
            f"Var. acumulada Dim1+Dim2 = {cum2_pct:.1f}%",
            transform=ax_mca2.transAxes, va="top", ha="left",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7)
        )
    fig_mca2.tight_layout()
    st.pyplot(fig_mca2, clear_figure=True, use_container_width=False)

# =========================
# Fila 2 (centrada): Heatmap de "loadings" AGRANDADO con valores
# =========================
st.subheader(f"MCA — Heatmap de loadings (primeras {n_components_mca} dimensiones {LABEL_MCA} varianza)")

col_coords_df = get_col_coords(mca_pr, data_cat)
if col_coords_df is None:
    st.info("No se pudieron obtener coordenadas de columnas para el heatmap.")
else:
    # Construir índice "variable=nivel" cuando hay MultiIndex
    if isinstance(col_coords_df.index, pd.MultiIndex) and col_coords_df.index.nlevels >= 2:
        idx_names = [f"{i[0]}={i[1]}" for i in col_coords_df.index]
    else:
        idx_names = col_coords_df.index.astype(str).tolist()

    loadings_mca = col_coords_df.iloc[:, :n_components_mca].copy()
    loadings_mca.index = idx_names

    # Tamaño dinámico AGRANDADO y anotaciones SIEMPRE visibles
    n_vars, n_dims = loadings_mca.shape
    fig_width  = min(24, max(12, 0.45 * n_dims + 12))   # ancho crece con #dimensiones
    fig_height = min(22, max(10, 0.30 * n_vars + 8))    # alto crece con #variables

    fig_mca3, ax_mca3 = plt.subplots(figsize=(fig_width, fig_height), dpi=120)
    sns.heatmap(
        loadings_mca,
        cmap="viridis", center=0,
        annot=True, fmt=".2f",        # ← valores en cada celda
        annot_kws={"size": 8},
        cbar_kws={"label": "Coordenada (loading)"}
    )
    ax_mca3.set_xlabel("Dimensiones MCA")
    ax_mca3.set_ylabel("Categorías (variable=nivel)")
    ax_mca3.set_title(f"Loadings MCA (hasta {LABEL_MCA} de varianza)")
    ax_mca3.set_xticklabels(ax_mca3.get_xticklabels(), rotation=45, ha="right")
    ax_mca3.set_yticklabels(ax_mca3.get_yticklabels(), rotation=0)
    fig_mca3.tight_layout()

    # Centrar el heatmap
    _, col_mid, _ = st.columns([1, 6, 1])
    with col_mid:
        st.pyplot(fig_mca3, clear_figure=True, use_container_width=False)


st.markdown("""
- *MCA – Varianza acumulada de variables categóricas:* La curva indica que con 34 dimensiones se supera el 90% de varianza explicada. La información está dispersa entre varias categorías, no hay ninguna que predomine. Usar 34 dimensiones reduce la explosión por one-hot manteniendo la estructura de asociaciones.

- *Dispersión de las dos primeras dimensiones:* La nube de puntos de Dim1 vs Dim2 muestra alto solapamiento entre healthy y diseased. Las dos primeras dimensiones no separan por sí solas las clases; se requiere el conjunto completo de dimensiones MCA (o modelos no lineales/supervisados) para capturar mejor los patrones.

- *Heatmap de loadings (34 dim. al 90%):* Las cargas están repartidas: variables como ocupación, tipo de trabajo, nivel educativo, tipo de dieta, hábito de fumar, uso de dispositivos, seguro/acceso a salud, apoyo en salud mental, exposición al sol, antecedentes familiares, etc., contribuyen en distintas dimensiones. Esto confirma correlaciones y asociaciones cruzadas (no dominadas por una sola categoría), y justifica el uso de MCA para condensarlas.
""")


# =========================
# Combinar PCA (num) + MCA (cat) para ML y mostrar preview
# =========================
st.markdown("## Dataset combinado: PCA + MCA (entrenamiento)")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import prince

# --- Fallback: si no existe un split global, créalo SIN muestreo (80/20 estratificado) ---
if 'X_train' not in locals() or 'y_train' not in locals():
    def make_train_split(df, seed=42):
        drop_cols = ['survey_code', 'bmi_estimated', 'bmi_scaled', 'bmi_corrected']
        X_ = df.drop(columns=[c for c in drop_cols + ['target'] if c in df.columns], errors='ignore')
        if 'target' not in df.columns:
            st.warning("No se encontró la columna `target` en los datos.")
            st.stop()
        y_ = df['target']
        return train_test_split(X_, y_, train_size=0.80, stratify=y_, random_state=seed)
    X_train, X_test, y_train, y_test = make_train_split(data)

# --- Helper para inercia (MCA) compatible con versiones de prince ---
def mca_inertia_array(model):
    if hasattr(model, "explained_inertia_") and model.explained_inertia_ is not None:
        arr = np.asarray(model.explained_inertia_, dtype=float)
        if arr.size:
            return arr
    if hasattr(model, "eigenvalues_") and model.eigenvalues_ is not None:
        ev = np.asarray(model.eigenvalues_, dtype=float); s = ev.sum()
        return ev / s if s > 0 else np.zeros_like(ev)
    if hasattr(model, "singular_values_") and model.singular_values_ is not None:
        sv = np.asarray(model.singular_values_, dtype=float); ev = sv**2; s = ev.sum()
        return ev / s if s > 0 else np.zeros_like(ev)
    return np.array([])

# =========================
# PCA (numéricas)
# =========================
UMBRAL_VAR_LOCAL = UMBRAL_VAR if 'UMBRAL_VAR' in locals() else 0.90

num_cols_pca = X_train.select_dtypes(include=np.number).columns.tolist()
if len(num_cols_pca) > 0:
    num_imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_num_train = num_imputer.fit_transform(X_train[num_cols_pca])
    X_num_train_std = scaler.fit_transform(X_num_train)

    pca_all = PCA().fit(X_num_train_std)
    cum_var = np.cumsum(pca_all.explained_variance_ratio_)
    n_comp_pca = int(np.searchsorted(cum_var, UMBRAL_VAR_LOCAL) + 1)

    X_train_pca_optimal = PCA(n_components=n_comp_pca).fit_transform(X_num_train_std)
    X_train_pca_df = pd.DataFrame(
        X_train_pca_optimal,
        index=X_train.index,
        columns=[f"PC{i+1}" for i in range(n_comp_pca)]
    )
else:
    X_train_pca_df = pd.DataFrame(index=X_train.index)  # vacío si no hay numéricas

# =========================
# MCA (categóricas) 
# =========================
UMBRAL_MCA_LOCAL = UMBRAL_MCA if 'UMBRAL_MCA' in locals() else 0.90

cat_cols_mca = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
if len(cat_cols_mca) > 0:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_train_cat = pd.DataFrame(
        cat_imputer.fit_transform(X_train[cat_cols_mca]),
        columns=cat_cols_mca,
        index=X_train.index
    ).astype(str)

    mca_pr = prince.MCA(
        n_components=min(50, len(cat_cols_mca) * 3),
        n_iter=5, copy=True, check_input=True, random_state=42
    ).fit(X_train_cat)

    inertia = mca_inertia_array(mca_pr)
    cum_inertia = np.cumsum(inertia)
    n_comp_mca = int(np.searchsorted(cum_inertia, UMBRAL_MCA_LOCAL) + 1)

    row_coords_train = mca_pr.row_coordinates(X_train_cat).to_numpy()[:, :n_comp_mca]
    X_train_mca_df = pd.DataFrame(
        row_coords_train,
        index=X_train.index,
        columns=[f"MCA{i+1}" for i in range(n_comp_mca)]
    )
else:
    X_train_mca_df = pd.DataFrame(index=X_train.index)  # vacío si no hay categóricas

# =========================
# Concatenar PCA + MCA
# =========================
X_train_combined = pd.concat([X_train_pca_df, X_train_mca_df], axis=1)

st.info(f"DataFrame combinado PCA + MCA (train): **{X_train_combined.shape[0]:,} filas × {X_train_combined.shape[1]:,} columnas**")
st.dataframe(X_train_combined.head(50), use_container_width=True)


st.markdown("## Resultados de la selección de Modelo")

from PIL import Image

# Cargar la imagen
image = Image.open('C:/Users/juanr/OneDrive/Desktop/MaestriaML/Resultados_modelos.png')

# Mostrar la imagen
st.image(image, caption="Resultados Modelos", use_column_width=True)

st.markdown("""
Los resultados muestran los rendimientos de diferentes modelos evaluados con los siguientes criterios:

### Mejor modelo según F1-macro en CV: 
 
ExtraTrees tiene el mejor rendimiento de acuerdo con la métrica F1-macro en la validación cruzada (CV) con un valor de 0.5032.

### Métricas adicionales:

- **Accuracy**: ExtraTrees tiene un accuracy de 59.04% en el conjunto de prueba.
- **F1-macro en prueba**: Un valor de 0.5006, lo que indica un rendimiento moderado.
- **Precision y Recall**: La precisión y el recall para F1-macro son 0.5009, que muestran que el modelo está balanceado en términos de precisión y recuperación en el conjunto de prueba.
- **Rendimiento en el conjunto de entrenamiento**: ExtraTrees tiene un rendimiento decente de 63.44% de accuracy y un F1-macro de 0.5568.
""")

# Cargar la imagen
image = Image.open('C:/Users/juanr/OneDrive/Desktop/MaestriaML/Hiperparametros.png')

# Mostrar la imagen
st.image(image, caption="Resultados Modelos", use_column_width=True)

st.markdown("## Aplicación del modelo escogido - ExtraTrees")

# --------- Modelos y espacios de búsqueda ---------

# 1. Define los hiperparámetros sin el prefijo 'classifier__'
best_params_extra_trees = {
    'n_estimators': 53,
    'max_depth': 8,
    'max_features': 'log2',
    'min_samples_leaf': 6,
    'min_samples_split': 11,
    'bootstrap': False
}

# 2. Asegúrate de que ExtraTreesClassifier reciba los parámetros correctos
pipe_extra_trees = ImbPipeline(steps=[
    ('preprocessor', preprocessor),   # Usa el preprocessor ya definido
    ('smote', SMOTE(random_state=42)),
    ('pca', PCA()),                   # Por defecto no reduce (n_components=None)
    ('classifier', ExtraTreesClassifier(random_state=42, **best_params_extra_trees)),  # Aplicamos los mejores hiperparámetros
])

# Entrenamiento y evaluación
pipe_extra_trees.fit(X_train, y_train)

# Predicciones
y_pred_test_extra_trees = pipe_extra_trees.predict(X_test)
y_pred_train_extra_trees = pipe_extra_trees.predict(X_train)

# Reportes
rpt_test_extra_trees = classification_report(y_test, y_pred_test_extra_trees, output_dict=True, zero_division=0)
rpt_train_extra_trees = classification_report(y_train, y_pred_train_extra_trees, output_dict=True, zero_division=0)

# Mostrar resultados en Streamlit
st.subheader("Evaluación del modelo ExtraTrees con hiperparámetros óptimos")
st.markdown(f"**Accuracy (Test):** {rpt_test_extra_trees['accuracy']:.4f}")
st.markdown(f"**F1-macro (Test):** {rpt_test_extra_trees['macro avg']['f1-score']:.4f}")
st.markdown(f"**Precision-m (Test):** {rpt_test_extra_trees['macro avg']['precision']:.4f}")
st.markdown(f"**Recall-m (Test):** {rpt_test_extra_trees['macro avg']['recall']:.4f}")
st.markdown(f"**Accuracy (Train):** {rpt_train_extra_trees['accuracy']:.4f}")
st.markdown(f"**F1-macro (Train):** {rpt_train_extra_trees['macro avg']['f1-score']:.4f}")
st.markdown(f"**Precision-m (Train):** {rpt_train_extra_trees['macro avg']['precision']:.4f}")
st.markdown(f"**Recall-m (Train):** {rpt_train_extra_trees['macro avg']['recall']:.4f}")

st.markdown("""
El modelo ExtraTrees ajustado con los hiperparámetros óptimos muestra un rendimiento moderado. En el conjunto de prueba alcanzó un accuracy del 59%, con valores de F1-macro, precisión y recall cercanos a 0.50, lo que indica que logra un desempeño apenas superior al azar en la clasificación, con un equilibrio limitado entre clases. Estos resultados sugieren que, aunque el modelo es capaz de identificar patrones, su capacidad de generalización es reducida y no logra discriminar con alta efectividad entre las categorías.

En el conjunto de entrenamiento el desempeño fue ligeramente superior, con un accuracy del 63% y un F1-macro de 0.55, acompañado de una precisión y recall de valores similares. Esto evidencia que el modelo conserva cierta coherencia entre entrenamiento y prueba, evitando un sobreajuste marcado, aunque todavía presenta limitaciones en su capacidad predictiva. En conjunto, los resultados reflejan que ExtraTrees ofrece un rendimiento estable pero con margen de mejora, lo que hace recomendable explorar técnicas adicionales de balanceo de clases, optimización de variables o modelos alternativos.
""")

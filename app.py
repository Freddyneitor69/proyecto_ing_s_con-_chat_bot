import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile
import os
import json
import textwrap

# Optional imports for chatbot backends; we'll test availability at runtime
try:
    import openai
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

try:
    from transformers import pipeline as hf_pipeline
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

# If you prefer to hard-code your OpenAI key inside this file, set it here.
# WARNING: hard-coding secrets in source is insecure. Prefer environment variables.
# To embed a key, replace None with your key string (not recommended):
DEFAULT_OPENAI_API_KEY = None
from utils import run_model_metrics_subprocess, predict_and_save_subprocess
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix

# Importar las funciones desde nuestro archivo utils
from utils import load_data, load_model

# Plotly default template and color mapping (will switch per theme)
PALETTE = {'Yes': '#E57373', 'No': "#09B743"}

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Smart Business", # Título para la pestaña del navegador
    page_icon="logo.png",      # Icono para la pestaña del navegador
    layout="wide"
)

# --- Carga de Datos y Modelos ---
df = load_data('data/telco_churn.csv')
df_test = load_data('data/test_data.csv')
if df is None and df_test is None:
    st.stop()

model_files = {
    "Random Forest (Pipeline Completo)": "models/rf_pipeline.joblib",
    "Stacking (con PCA)": "models/stack_pca.joblib",
    "Stacking (Features Seleccionadas)": "models/stack_selected.joblib"
}

# --- Configuración de usuario / Tema (persistida en session_state) ---
def _init_settings():
    ss = st.session_state
    ss.setdefault('theme', 'Dark')  # Light or Dark (default to Dark as requested)
    ss.setdefault('primary_color', '#00B0F0')
    ss.setdefault('layout', 'wide')  # wide or centered
    ss.setdefault('show_feature_importance', True)
    ss.setdefault('default_threshold', 0.70)
    ss.setdefault('enable_proba', True)
    # Elegir un modelo por defecto si no existe
    ss.setdefault('default_model', list(model_files.keys())[0])


_init_settings()

# Hide the legacy sidebar and free up screen real estate; users will use the Config tab
_HIDE_SIDEBAR_CSS = """
<style>
/* App background and surface colors */
:root { --bg-main: #1A1D24; --surface: #252A33; --text-main: #F0F2F5; --text-secondary: #A0A8B5; }
body { background-color: var(--bg-main); color: var(--text-main); }
div.block-container { background-color: transparent; }
/* Style the main content cards / sections */
section[data-testid='stBlock'] > div[role='region'] { background-color: rgba(37,42,51,0.6); }
/* Hide Streamlit sidebar reliably */
[data-testid='stSidebar'] { display: none !important; }
/* Reduce left margin introduced by sidebar */
.css-1d391kg { margin-left: 0px !important; }
.stButton>button { background-color: transparent !important; color: var(--text-main) !important; }
/* Small helper text style */
.config-note { color: var(--text-secondary); font-size:12px }
</style>
"""
st.markdown(_HIDE_SIDEBAR_CSS, unsafe_allow_html=True)


# Aplicar ajustes visuales simples inyectando CSS según el tema seleccionado
def _apply_theme_css():
    theme = st.session_state.get('theme', 'Dark')
    primary = st.session_state.get('primary_color', '#00B0F0')
    if theme == 'Dark':
        px.defaults.template = 'plotly_dark'
        css = f"""
        <style>
        :root {{ --bg-main: #1A1D24; --surface: #252A33; --text-main: #F0F2F5; --text-secondary: #A0A8B5; --muted: #8A8F98; --primary: {primary}; }}
        body {{ background-color: var(--bg-main); color: var(--text-main); }}
        .block-container {{ background-color: transparent; }}
        section[data-testid='stBlock'] > div[role='region'] {{ background-color: var(--surface); border-radius:8px; padding: 10px; }}
    /* Ensure headings and titles are bold and grey */
    h1, h2, h3, [role='heading'] {{ color: var(--text-secondary) !important; font-weight:700 !important; }}
    /* Secondary text */
    p, label, .config-note {{ color: var(--text-secondary) !important; }}
    .stMetricValue {{ color: var(--text-main) !important; }}
    .stButton>button {{ background-color: transparent !important; color: var(--text-main) !important; border: 1px solid rgba(255,255,255,0.06); }}
    /* Plotly transparent backgrounds so underlying surface shows and text color for ticks/legend */
    .js-plotly-plot .plotly .main-svg, .plotly .legend text, .plotly .xtick text, .plotly .ytick text {{ fill: var(--text-secondary) !important; }}
    .stPlotlyChart > div > div > div {{ background: transparent !important; }}
        </style>
        """
    else:
        px.defaults.template = 'plotly_white'
        css = f"""
        <style>
        :root {{ --bg-main: #ffffff; --surface: #f7f7f7; --text-main: #111827; --text-secondary: #6B7280; --primary: {primary}; }}
        body {{ background-color: var(--bg-main); color: var(--text-main); }}
        section[data-testid='stBlock'] > div[role='region'] {{ background-color: var(--surface); border-radius:8px; padding:10px; }}
        h1, h2, h3 {{ color: var(--text-main) !important; font-weight:700 !important; }}
        .stButton>button {{ background-color: var(--primary) !important; color: white !important; }}
        .stPlotlyChart > div > div > div {{ background: transparent !important; }}
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


# If a widget 'ui_theme' exists from previous runs, sync it into the theme used for CSS
if 'ui_theme' in st.session_state:
    st.session_state['theme'] = st.session_state.get('ui_theme', st.session_state.get('theme', 'Dark'))

_apply_theme_css()


# -------------------------
# Chatbot helpers
# -------------------------
@st.cache_resource
def get_chat_backend():
    """Return a tuple (provider, model_obj_or_module).
    provider is one of: 'openai', 'hf', 'fallback'.
    """
    # Prefer OpenAI if API key available and openai is installed
    # Prefer a key stored in the user's session, then the environment variable OPENAI_API_KEY.
    # We avoid using any hard-coded DEFAULT_OPENAI_API_KEY to ensure secrets are not in source.
    openai_key = st.session_state.get('openai_api_key') or os.environ.get('OPENAI_API_KEY')
    if openai_key and _HAS_OPENAI:
        try:
            # Initialize OpenAI client with the API key
            client = openai.OpenAI(api_key=openai_key)
            return ('openai', client)
        except Exception:
            pass

    # Then try HuggingFace transformers pipeline if available
    if _HAS_TRANSFORMERS:
        try:
            # Use a small local text-generation model by default (gpt2)
            pipe = hf_pipeline('text-generation', model='gpt2')
            return ('hf', pipe)
        except Exception:
            pass

    # Fallback: no external model available
    return ('fallback', None)


def _summarize_dataframe_for_prompt(df, max_rows=100, max_chars=10000):
    """Create a simple context with the first 100 rows of the DataFrame for the AI model."""
    if df is None or df.empty:
        return "Dataset: (empty)\n"
    
    # Remove customerID for cleaner sample if present
    sample_df = df.drop('customerID', axis=1) if 'customerID' in df.columns else df
    sample_csv = sample_df.head(max_rows).to_csv(index=False)
    
    if len(sample_csv) > max_chars:
        sample_csv = sample_csv[:max_chars] + '\n...truncated'
    
    return f"Dataset sample (first {max_rows} rows):\n{sample_csv}"


def run_chat(prompt: str, df_context: pd.DataFrame | None = None, max_tokens: int = 500) -> str:
    """Run the chat model (OpenAI GPT-4o mini preferred) with strict focus on dataset analysis."""
    provider, model_obj = get_chat_backend()
    
    # Enhanced system prompt to ensure model stays focused on dataset analysis
    if df_context is not None and not df_context.empty:
        dataset_context = _summarize_dataframe_for_prompt(df_context)
        system_prompt = f"""You are a data analyst specializing in customer retention for telecom companies. Here is the data:
{dataset_context}

Answer questions based only on this dataset. You can provide insights on customer churn, retention strategies, and analysis of the telecom data. If the question is not related to the data, say "I can only analyze the provided telcom dataset"."""

        user_message = prompt

    else:
        return "No hay dataset cargado. Por favor carga un dataset de telecomunicaciones para que pueda realizar el análisis."

    if provider == 'openai':
        try:
            # Use GPT-4o for advanced analysis
            response = model_obj.chat.completions.create(
                model="gpt-4o-mini",  # Changed to gpt-4o-mini for better availability
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=max_tokens,
                temperature=0.1,  # Low temperature for consistent, factual responses
                top_p=0.9
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Error OpenAI] {e}"

    elif provider == 'hf':
        try:
            # Enhanced prompt for local models
            full_prompt = f"{system_prompt}\n\n{user_message}\n\nRespuesta:"
            output = model_obj(full_prompt, 
                             max_length=len(full_prompt.split()) + max_tokens, 
                             do_sample=True, 
                             temperature=0.7,
                             pad_token_id=50256)
            
            if isinstance(output, list) and len(output) > 0 and 'generated_text' in output[0]:
                generated = output[0]['generated_text']
                # Extract only the response part
                if "Respuesta:" in generated:
                    return generated.split("Respuesta:")[-1].strip()
                return generated.strip()
            return str(output)
        except Exception as e:
            return f"[Error HF] {e}"

    else:
        # No backend available
        return "No hay un modelo de IA disponible. Por favor configura una API key de OpenAI o instala transformers para usar un modelo local."


# Asegurar que la selección de modelo en la UI use el modelo por defecto configurado
if 'model_pred' not in st.session_state:
    st.session_state['model_pred'] = st.session_state.get('default_model')

# --- Preparación de Datos para Métricas ---
X_test = df_test.drop('Churn', axis=1)
y_test = df_test['Churn'].map({'Yes': 1, 'No': 0})


# --- UI Principal ---

# --- TÍTULO CON LOGO ---
col1, col2, col3 = st.columns([1, 6, 1]) # Crea columnas para el logo, título y acciones (gear)

with col1:
    st.image("logo.png", width=180) # Muestra el logo

with col2:
    st.title("Smart Business: Dashboard de Decisión de Churn")
    st.markdown("Análisis inteligente para la retención de clientes.")

# gear icon in top-right to hint configuration tab
with col3:
    st.markdown("<div style='text-align:right'><a href='#' style='color:var(--text-main); text-decoration:none; font-size:20px;'>⚙️</a></div>", unsafe_allow_html=True)
    st.markdown("<div class='config-note' style='text-align:right'>Abrir pestaña ⚙️ Configuración</div>", unsafe_allow_html=True)
# --- Navegación por Pestañas ---
tab1, tab2, tab3, tab_chat, tab4 = st.tabs(["🔍 **Explorador y Predicción**", "💼 **Impacto de Negocio**", "📊 **Rendimiento de Modelos**", "💬 Chat Bot", "⚙️ Configuración"])


# =====================================================================================
# --- PESTAÑA 1: EXPLORADOR Y PREDICCIÓN ---
# =====================================================================================
with tab1:
    st.header("Análisis Exploratorio y Predicción Individual")

    # --- KPIs Estáticos ---
    st.subheader("Métricas Clave del Segmento")
    kpi1, kpi2, kpi3 = st.columns([1,1,1])
    churn_rate_yes = df['Churn'].value_counts(normalize=True).get('Yes', 0)
    # compute a simple baseline: churn rate for short-tenure customers (proxy trend)
    if 'tenure' in df.columns:
        short = df[df['tenure'] <= 12]
        baseline = short['Churn'].value_counts(normalize=True).get('Yes', churn_rate_yes)
    else:
        baseline = churn_rate_yes
    churn_delta = churn_rate_yes - baseline
    # delta_color: higher churn is bad (use inverse)
    kpi1.metric("Tasa de Fuga (Churn)", f"{churn_rate_yes:.2%}", delta=f"{churn_delta:+.2%}", delta_color="inverse", help="Comparado con clientes con tenure <= 12 meses")

    avg_monthly_charge = df['MonthlyCharges'].mean()
    # show a small delta vs median monthly charges
    median_charge = df['MonthlyCharges'].median()
    kpi2.metric("Cargo Mensual Promedio", f"${avg_monthly_charge:.2f}", delta=f"${avg_monthly_charge - median_charge:,.2f}", help="Delta vs mediana del segmento")

    # Mini-sparkline: churn rate por tenure para dar contexto temporal (si existe tenure)
    if 'tenure' in df.columns:
        churn_by_tenure = df.groupby('tenure')['Churn'].apply(lambda s: (s=='Yes').mean()).reset_index(name='churn_rate')
        # show a compact line chart under the KPIs
        with kpi3:
            st.markdown("**Tendencia (por tenure)**")
            st.line_chart(churn_by_tenure.set_index('tenure')['churn_rate'], height=80)
    else:
        kpi3.metric("Contexto", "Sin serie temporal", help="No hay columna 'tenure' para generar tendencia")

    # --- Resultados de la sección EDA (Visualizaciones Específicas) ---
    st.subheader("Visualizaciones Específicas")
    row1_col1, row1_col2 = st.columns(2)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols.remove('customerID')
    
    with row1_col1:
        st.markdown("##### 📈 Distribución de una Variable")
        dist_var = st.selectbox("Variable:", options=numeric_cols + categorical_cols, key="dist_var")
        if dist_var in numeric_cols:
            fig = px.histogram(df, x=dist_var, color="Churn", marginal="box", color_discrete_map=PALETTE)
            fig.update_layout(title_text=f"Distribución de {dist_var}", xaxis_title=dist_var, yaxis_title='Conteo', plot_bgcolor='rgba(0,0,0,0)')
        else:
            # For categorical variables show 100% stacked bars with churn proportions per category
            counts = df.groupby([dist_var, 'Churn']).size().reset_index(name='count')
            total = counts.groupby(dist_var)['count'].transform('sum')
            counts['pct'] = counts['count'] / total
            # pivot to ensure both classes exist for each category
            pivot = counts.pivot(index=dist_var, columns='Churn', values='pct').fillna(0)
            pivot = pivot.reset_index()
            # Melt for plotting stacked bars
            melt = pivot.melt(id_vars=[dist_var], value_vars=[c for c in pivot.columns if c != dist_var], var_name='Churn', value_name='pct')
            # Use requested desaturated colors for dark mode
            fig = px.bar(melt, x=dist_var, y='pct', color='Churn', color_discrete_map={'Yes': PALETTE['Yes'], 'No': PALETTE['No']}, text=melt['pct'].apply(lambda v: f"{v:.0%}"))
            fig.update_layout(barmode='stack', title_text=f"Composición de Churn por {dist_var}", yaxis_tickformat='%')
            fig.update_traces(hovertemplate=f"%{{x}}<br>%{{y:.1%}} of category")
        st.plotly_chart(fig, use_container_width=True)

    with row1_col2:
        st.markdown("##### 📦 Box Plots (Numérico vs. Categórico)")
        cat_box = st.selectbox("Variable Categórica:", options=categorical_cols, key="cat_box", index=categorical_cols.index('Contract'))
        num_box = st.selectbox("Variable Numérica:", options=numeric_cols, key="num_box", index=numeric_cols.index('MonthlyCharges'))
        # Enhanced boxplot: show points and clear axis labels, ensure visibility on dark bg
        fig_box = px.box(df, x=cat_box, y=num_box, color="Churn", points='all', color_discrete_map={'Yes': PALETTE['Yes'], 'No': PALETTE['No']})
        fig_box.update_traces(marker=dict(opacity=0.8, line=dict(width=0.5, color='rgba(0,0,0,0.2)')))
        fig_box.update_layout(title_text=f"{num_box} por {cat_box}", xaxis_title=cat_box, yaxis_title=num_box, plot_bgcolor='rgba(0,0,0,0)')
        # Ensure median and box lines stand out
        fig_box.update_traces(boxmean=True)
        st.plotly_chart(fig_box, use_container_width=True)

    st.divider()
    # --- Sección de Clasificación (Predicción) ---
    st.header("Predicción de Churn para un Nuevo Cliente 🔮")
    model_name_pred = st.selectbox("Elige el pipeline de predicción:", options=list(model_files.keys()), key="model_pred")
    model_path_pred = model_files[model_name_pred]
    # Mostrar columnas esperadas por el pipeline (útil para depurar transformación de entrada)
    if st.button("Mostrar columnas esperadas del modelo", key="inspect_cols_btn"):
        try:
            from utils import inspect_model_columns_subprocess
            expected_cols = inspect_model_columns_subprocess(model_path_pred)
            if expected_cols:
                st.info(f"El modelo espera las siguientes columnas ({len(expected_cols)}):")
                st.write(expected_cols)
            else:
                st.info("No se pudieron inferir las columnas esperadas del modelo.")
        except Exception as e:
            st.error(f"No se pudo inspeccionar el modelo: {e}")
    
    with st.form("prediction_form"):
        input_data = {}
        form_cols = st.columns(3)
        features = df.drop(columns=['customerID', 'Churn']).columns
        for i, col in enumerate(features):
            with form_cols[i % 3]:
                if df[col].dtype == "object":
                    input_data[col] = st.selectbox(label=col, options=df[col].unique().tolist(), key=f"form_{col}")
                elif col == 'SeniorCitizen':
                    input_data[col] = st.selectbox(label=col, options=[0, 1], key=f"form_{col}")
                else:
                    input_data[col] = st.number_input(label=col, value=float(df[col].median()), key=f"form_{col}")
        submit_button = st.form_submit_button(label="Predecir Churn")

    if submit_button:
        input_df = pd.DataFrame([input_data])
        # initialize prediction variables to safe defaults
        prediction = None
        prob_churn = None
        churn_status = None
        # Always write the input to a temporary CSV and call the subprocess helper.
        import tempfile, os
        fd, tmp_in = tempfile.mkstemp(prefix='input_', suffix='.csv')
        os.close(fd)
        input_df.to_csv(tmp_in, index=False)
        tmp_out = None
        try:
            tmp_out = predict_and_save_subprocess(model_path_pred, tmp_in)
        except Exception as e:
            st.warning(f"No se pudo ejecutar la predicción: {e}")
        finally:
            try:
                os.remove(tmp_in)
            except Exception:
                pass

        if tmp_out:
            preds_df = pd.read_csv(tmp_out)
            # keep the CSV on disk until we've rendered results to the UI
            # (Streamlit may reference content while rendering; deleting too early
            # can cause front-end DOM removal errors in some environments)

            if 'prediction' in preds_df.columns:
                prediction = preds_df['prediction'].iloc[0]
                prob_churn = preds_df['prediction_proba'].iloc[0] if 'prediction_proba' in preds_df.columns else None
                churn_status = "Sí (Fuga)" if int(prediction) == 1 else "No (Permanece)"
            else:
                st.warning('El proceso de predicción no devolvió una columna `prediction`.')
                prediction = None
                prob_churn = None

        st.subheader("Resultado de la Predicción:")
        if prediction is not None:
            if churn_status == "Sí (Fuga)":
                st.error(f"El cliente probablemente hará Churn: **{churn_status}**")
            else:
                st.success(f"El cliente probablemente se quedará: **{churn_status}**")
            if prob_churn is not None:
                st.metric(f"Probabilidad de Fuga (Modelo: {model_name_pred})", f"{prob_churn:.2%}")
            else:
                st.info("Probabilidad no disponible.")
        else:
            st.warning("No se pudo generar una predicción para los datos ingresados. Revisa los logs o intenta nuevamente.")

        # --- Automatic Recommendation ---
        # Guardar y validar la probabilidad antes de comparar (evitar TypeError si es None)
        try:
            prob_val = float(prob_churn) if prob_churn is not None else None
        except Exception:
            prob_val = None

        if prob_val is not None and prob_val > 0.70:
            st.warning("🚨 **ALTO RIESGO DE FUGA DETECTADO**")
            st.info("Acción sugerida: Contactar al cliente proactivamente. Ofrecer un descuento temporal, un nuevo servicio de valor agregado o una mejora en su plan actual para aumentar la retención.")
        elif prob_val is None:
            # Si no hay probabilidad disponible, mostrar información opcional
            st.info("No hay una probabilidad disponible para generar una recomendación automática.")

        # CLEANUP: borrar archivo temporal de salida sólo después de renderizar todo
        try:
            if tmp_out and os.path.exists(tmp_out):
                os.remove(tmp_out)
        except Exception:
            # no es crítico; ignorar errores de borrado
            pass


# =====================================================================================
# --- PESTAÑA 2: IMPACTO DE NEGOCIO ---
# =====================================================================================
with tab2:
    st.header("Simulación del Impacto en el Negocio")
    st.markdown("Analiza el riesgo financiero y el potencial de ahorro basado en las predicciones del modelo.")

    model_name_biz = st.selectbox("Selecciona un modelo para la simulación:", options=list(model_files.keys()), key="model_biz")
    model_path_biz = model_files[model_name_biz]

    try:
        # Ejecutar la predicción en un proceso separado y leer el CSV con la columna 'prediction'
        tmp_preds = predict_and_save_subprocess(model_path_biz, 'data/telco_churn.csv')
        df_biz = pd.read_csv(tmp_preds)
    except Exception as e:
        st.warning(f"No se pudo ejecutar la predicción en el modelo seleccionado: {e}")
        df_biz = df.copy()
        df_biz['prediction'] = 0

    # --- Business logic (ejecutar tanto si la predicción fue exitosa como si usamos fallback) ---
    customers_at_risk = df_biz[df_biz['prediction'] == 1]
    customers_loyal = df_biz[df_biz['prediction'] == 0]

    # --- Business KPIs ---
    st.subheader("KPIs de Riesgo y Oportunidad")
    kpi_biz1, kpi_biz2, kpi_biz3 = st.columns(3)

    pct_at_risk = len(customers_at_risk) / len(df_biz) if len(df_biz) > 0 else 0
    kpi_biz1.metric("% de Clientes en Riesgo de Fuga", f"{pct_at_risk:.2%}")

    revenue_at_risk = customers_at_risk['MonthlyCharges'].sum() if 'MonthlyCharges' in customers_at_risk.columns else 0
    kpi_biz2.metric("Ingresos Mensuales en Riesgo", f"${revenue_at_risk:,.2f}")

    # --- NUEVO: KPI de Valor Promedio del Cliente (ARPU) ---
    arpu_at_risk = customers_at_risk['MonthlyCharges'].mean() if (not customers_at_risk.empty and 'MonthlyCharges' in customers_at_risk.columns) else 0
    arpu_loyal = customers_loyal['MonthlyCharges'].mean() if (not customers_loyal.empty and 'MonthlyCharges' in customers_loyal.columns) else 0
    kpi_biz3.metric(
        label="Valor Promedio Cliente en Riesgo",
        value=f"${arpu_at_risk:,.2f}",
        delta=f"${arpu_at_risk - arpu_loyal:,.2f} vs. leales",
        delta_color="inverse",
        help="Un delta negativo significa que los clientes en riesgo pagan menos que los leales. Un delta positivo significa que pagan más y son más valiosos."
    )

    st.divider()

    st.subheader("Simulación de Campaña de Retención")
    sim_col1, sim_col2 = st.columns([2, 1])

    with sim_col1:
        st.markdown("#### Parámetros de la Simulación")
        retention_pct = st.slider("Selecciona el % de clientes en riesgo que podrías retener:", 1, 100, 10, key="retention_slider")
        
        # --- NUEVO: KPI de Costo de Retención ---
        cost_per_customer = st.number_input("Costo de retención por cliente ($):", min_value=0, value=15, step=1, key="cost_input")
        
    # --- Cálculos de la Simulación ---
    customers_retained = int(len(customers_at_risk) * (retention_pct / 100.0))
    potential_savings = customers_retained * arpu_at_risk # Más preciso que usar el ingreso total
    total_campaign_cost = customers_retained * cost_per_customer
    
    # --- NUEVO: KPI de Retorno de la Inversión (ROI) ---
    roi = (potential_savings - total_campaign_cost) / total_campaign_cost if total_campaign_cost > 0 else 0
    
    with sim_col2:
        st.markdown("#### Resultados Estimados")
        st.metric(
            label="💰 Ahorro Mensual Potencial",
            value=f"${potential_savings:,.2f}"
        )
        st.metric(
            label="💸 Costo Total de la Campaña",
            value=f"${total_campaign_cost:,.2f}"
        )
        st.metric(
            label="📈 ROI de la Campaña",
            value=f"{roi:.2%}",
            help="(Ahorro - Costo) / Costo"
        )

    # Visualización de la simulación
    fig_sim = go.Figure(go.Bar(
        x=['Ahorro Potencial', 'Costo de Campaña'],
        y=[potential_savings, total_campaign_cost],
        marker_color=['#2ECC71', '#E74C3C'],
        text=[f"${potential_savings:,.0f}", f"${total_campaign_cost:,.0f}"],
        textposition='auto'
    ))
    fig_sim.update_layout(title_text="Desglose Financiero de la Campaña de Retención", yaxis_title="Monto ($)")
    st.plotly_chart(fig_sim, use_container_width=True)

# =====================================================================================
# --- PESTAÑA 3: RENDIMIENTO DE MODELOS ---
# =====================================================================================
with tab3:
    st.header("Comparación y Análisis de Rendimiento de Modelos")
    
    # --- Metrics Comparison ---
    st.subheader("📊 Comparación de Métricas (sobre datos de Test)")
    
    metrics_data = []
    for model_name, model_path in model_files.items():
        try:
            metrics = run_model_metrics_subprocess(model_path, 'data/test_data.csv')
        except Exception as e:
            st.warning(f"No se pudieron calcular métricas para {model_name}: {e}")
            continue
        metrics_data.append({
            "Modelo": model_name,
            "Accuracy": metrics.get('Accuracy', None),
            "F1-Score": metrics.get('F1-Score', None),
            "AUC": metrics.get('AUC', None)
        })
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data).set_index("Modelo")
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen').format("{:.4f}"))
    else:
        st.warning("No se pudieron calcular métricas para ninguno de los modelos. Revisa los logs o ejecuta las predicciones en un proceso separado.")
        metrics_df = pd.DataFrame()

    st.divider()

    col_cm, col_fi = st.columns(2)

    with col_cm:
        # --- Confusion Matrix ---
        st.subheader("🔥 Matriz de Confusión")
        model_name_cm = st.selectbox("Selecciona un modelo para ver su matriz de confusión:", options=list(model_files.keys()), key="model_cm")
        model_path_cm = model_files[model_name_cm]
        try:
            # Ensure we run predictions on the same cleaned dataframe used by the app
            # (load_data() may drop rows with invalid TotalCharges, causing length mismatches)
            fd_tmp, tmp_test_path = tempfile.mkstemp(prefix='testdata_', suffix='.csv')
            os.close(fd_tmp)
            try:
                # write the cleaned df_test (as used above) to a temp file and predict on it
                df_test.to_csv(tmp_test_path, index=False)
                tmp_cm = predict_and_save_subprocess(model_path_cm, tmp_test_path)
                preds_df = pd.read_csv(tmp_cm)
            finally:
                # cleanup the temporary test file immediately (predictions are written to a separate tmp file)
                try:
                    if os.path.exists(tmp_test_path):
                        os.remove(tmp_test_path)
                except Exception:
                    pass

            # Sanity checks and alignment: ensure prediction column exists and is integer-typed
            if 'prediction' not in preds_df.columns:
                raise RuntimeError("El CSV de predicciones no contiene la columna 'prediction'.")
            y_pred_cm = preds_df['prediction'].astype(int)

            # Verify lengths match between cleaned y_test and predictions (they should after using df_test)
            if len(y_pred_cm) != len(y_test):
                # Try to align by customerID if available
                if 'customerID' in preds_df.columns and 'customerID' in df_test.columns:
                    merged = pd.merge(df_test.reset_index(), preds_df[['customerID', 'prediction']], on='customerID', how='left')
                    if merged['prediction'].isnull().any():
                        raise RuntimeError('No se pudo alinear las predicciones con el conjunto de test limpio (customerID mismatch).')
                    y_pred_cm = merged['prediction'].astype(int)
                else:
                    raise RuntimeError(f'Longitud de predicciones ({len(y_pred_cm)}) no coincide con conjunto de test ({len(y_test)}).')

            cm = confusion_matrix(y_test.astype(int), y_pred_cm)
        except Exception as e:
            st.warning(f"No se pudo generar la matriz de confusión para {model_name_cm}: {e}")
            cm = None
        if cm is None:
            st.info('Matriz de confusión no disponible para el modelo seleccionado.')
        else:
            fig_cm, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
            plt.xlabel('Predicción')
            plt.ylabel('Real')
            st.pyplot(fig_cm)

    with col_fi:
        # --- Feature Importance Plot ---
        st.subheader("⭐ Importancia de Features")
        try:
            fi_df = pd.read_csv('data/feature_importance_combined.csv')
        except Exception as e:
            st.warning(f"No se pudo abrir 'feature_importance_combined.csv': {e}")
        else:
            # Normalizar nombres de columnas comunes
            cols_lower = [c.lower() for c in fi_df.columns]
            if 'feature' in cols_lower and 'importance' in cols_lower:
                feat_col = fi_df.columns[cols_lower.index('feature')]
                imp_col = fi_df.columns[cols_lower.index('importance')]
            else:
                # Fallback: usar las dos primeras columnas asumidas como feature/importance
                feat_col, imp_col = fi_df.columns[0], fi_df.columns[1]

            # Asegurar que la importancia sea numérica y limpiar
            fi_df[imp_col] = pd.to_numeric(fi_df[imp_col], errors='coerce')
            fi_df = fi_df.dropna(subset=[imp_col, feat_col])

            if fi_df.empty:
                st.warning("El archivo no contiene datos válidos de importancia de features.")
            else:
                # Configurable: seleccionar top N
                max_n = len(fi_df)
                top_n = st.slider("Mostrar top N features", min_value=1, max_value=max_n, value=min(20, max_n), key="top_n_features")
                fi_sorted = fi_df.sort_values(by=imp_col, ascending=False).head(top_n)

                st.subheader("Importancia de Features")
                st.dataframe(fi_sorted.reset_index(drop=True))

                fig_fi = px.bar(
                    fi_sorted,
                    x=imp_col,
                    y=feat_col,
                    orientation='h',
                    title=f"Top {top_n} Features por Importancia",
                    color=imp_col,
                    color_continuous_scale='blues'
                )
                fig_fi.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False)
                st.plotly_chart(fig_fi, use_container_width=True)


        # =====================================================================================
        # --- PESTAÑA: CHAT BOT ---
        # =====================================================================================
        with tab_chat:
            st.header("💬 Chat Bot")
            st.markdown("Interactúa con un modelo de lenguaje para explorar datos o responder preguntas sobre el dataset de telecomunicaciones.")

            # Dataset is already loaded at app startup, ensure it's available
            if 'chat_df' not in st.session_state or st.session_state['chat_df'] is None or st.session_state['chat_df'].empty:
                st.session_state['chat_df'] = df.copy() if df is not None else pd.DataFrame()

            col_left, col_right = st.columns([3, 1])

            with col_left:
                prompt = st.text_area('Escribe tu prompt aquí', value='¿Cuáles son las estrategias de retención más efectivas basadas en los datos? Analiza los factores que influyen en el churn.', height=140)

                if st.button('Enviar', key='chat_send'):
                    with st.spinner('Generando respuesta...'):
                        try:
                            df_context = st.session_state.get('chat_df', None)
                            reply = run_chat(prompt, df_context)
                        except Exception as e:
                            reply = f'Error al generar respuesta: {e}'
                    st.write(reply)

            with col_right:
                st.markdown('### Dataset actual (preview)')
                cur = st.session_state.get('chat_df')
                if cur is None or cur.empty:
                    st.info('No hay dataset cargado.')
                else:
                    st.dataframe(cur.head(10))
                    st.markdown(f"- Filas: {cur.shape[0]} | Columnas: {cur.shape[1]}")
                    backend, _ = get_chat_backend()
                    st.markdown(f"**Backend disponible:** {backend}")
                    # Debug info: show whether an API key is set and its source (session or environment)
                    env_key = bool(os.environ.get('OPENAI_API_KEY'))
                    session_key = bool(st.session_state.get('openai_api_key'))
                    api_key_present = session_key or env_key
                    st.write(f"Debug: API Key configurada: {api_key_present}")
                    if api_key_present:
                        src = 'session' if session_key else 'environment'
                        st.success(f"API Key configurada (fuente: {src})")
                    else:
                        st.warning("No hay API Key configurada. Puedes pegarla en ⚙️ Configuración o exportar OPENAI_API_KEY en tu entorno.")

                st.markdown('---')
                st.markdown('### OpenAI API Key')
                # Provide clear instructions and do not indicate any key is stored in code
                st.info('Introduce la clave en ⚙️ Configuración o exporta la variable de entorno OPENAI_API_KEY. No dejes la clave en el código fuente.')



# =====================================================================================
# --- PESTAÑA 4: CONFIGURACIÓN (moved from sidebar) ---
# =====================================================================================
with tab4:
    st.header("⚙️ Configuración")
    st.markdown("Ajusta tema, modelos y preferencias de visualización. Los cambios se aplican en la sesión actual.")
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Apariencia")
        theme = st.selectbox('Tema', options=['Light', 'Dark'], index=0 if st.session_state['theme']=='Light' else 1, key='ui_theme')
        st.session_state['theme'] = theme
        primary = st.color_picker('Color primario', value=st.session_state['primary_color'], key='ui_primary')
        st.session_state['primary_color'] = primary
        layout_choice = st.selectbox('Layout', options=['wide', 'centered'], index=0 if st.session_state['layout']=='wide' else 1, key='ui_layout')
        st.session_state['layout'] = layout_choice

        st.subheader('Modelos y Predicción')
        options = list(model_files.keys())
        default_index = options.index(st.session_state.get('default_model')) if st.session_state.get('default_model') in options else 0
        default_model = st.selectbox('Modelo por defecto', options=options, index=default_index, key='ui_default_model')
        st.session_state['default_model'] = default_model
        enable_proba = st.checkbox('Mostrar probabilidades', value=st.session_state['enable_proba'], key='ui_enable_proba', help='Incluir columna de probabilidad en predicciones')
        st.session_state['enable_proba'] = enable_proba
        threshold = st.slider('Umbral (alto riesgo)', min_value=0.0, max_value=1.0, value=float(st.session_state['default_threshold']), step=0.01, key='ui_threshold')
        st.session_state['default_threshold'] = threshold

        st.subheader('Visualización')
        show_fi = st.checkbox('Mostrar FI por defecto', value=st.session_state['show_feature_importance'], key='ui_show_fi')
        st.session_state['show_feature_importance'] = show_fi

        # --- OpenAI API Key (manual entry into session) ---
        st.subheader('OpenAI API Key')
        st.markdown('No almacenes la API key en el código. Pégala temporalmente para esta sesión o exporta la variable de entorno OPENAI_API_KEY.')
        openai_temp = st.text_input('Pegar OpenAI API Key (solo para esta sesión)', type='password', key='openai_api_key_input')
        if st.button('Guardar API Key en sesión', key='save_openai_key'):
            if openai_temp:
                st.session_state['openai_api_key'] = openai_temp
                st.success('API Key guardada en la sesión (no se escribe en disco).')
            else:
                st.warning('Introduce una clave antes de guardar.')
        if st.button('Borrar API Key de sesión', key='clear_openai_key'):
            if 'openai_api_key' in st.session_state:
                st.session_state.pop('openai_api_key', None)
            st.info('API Key de sesión borrada.')

        if st.button('Restablecer configuración'):
            st.session_state['theme'] = 'Light'
            st.session_state['primary_color'] = '#00B0F0'
            st.session_state['layout'] = 'wide'
            st.session_state['show_feature_importance'] = True
            st.session_state['default_threshold'] = 0.70
            st.session_state['enable_proba'] = True
            st.session_state['default_model'] = list(model_files.keys())[0]
            st.experimental_rerun()

    with col_right:
        st.markdown('### Vista previa')
        st.write('Los cambios de tema y color se aplicarán inmediatamente en la sesión actual.')
        if st.session_state.get('theme') == 'Dark':
            st.image('logo.png', width=140)
        else:
            st.image('logo.png', width=120)


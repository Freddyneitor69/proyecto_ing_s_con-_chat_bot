# Dashboard Interactivo de Análisis y Predicción de Churn

Este proyecto es una aplicación web interactiva construida con Streamlit para analizar la fuga de clientes (churn) en una empresa de telecomunicaciones. El dashboard no solo permite explorar los datos de forma dinámica, sino que también integra modelos de Machine Learning para predecir qué clientes están en riesgo y simular el impacto financiero de la retención.[Image de un panel de control de análisis de datos de clientes]

# 📋 Características Principales
- Análisis Exploratorio Interactivo (EDA): Visualiza la distribución y relación entre las distintas características de los clientes a través de gráficos dinámicos que se actualizan según los filtros aplicados.
- Predicción Individual: Ingresa los datos de un nuevo cliente en un formulario y obtén una predicción instantánea sobre su probabilidad de fuga utilizando pipelines de Machine Learning autónomos.
- Simulación de Impacto de Negocio: Una pestaña dedicada a traducir las predicciones en KPIs financieros, como los ingresos mensuales en riesgo y el ahorro potencial al implementar campañas de retención.
- Comparación de Modelos: Evalúa y compara el rendimiento de tres pipelines de clasificación diferentes (Random Forest, Stacking con PCA y Stacking con Features Seleccionadas) a través de métricas clave (Accuracy, F1-Score, AUC) y matrices de confusión.
- Recomendaciones Automáticas: El sistema genera alertas y sugerencias de acción para clientes con una alta probabilidad de fuga.

# 🚀 Cómo Ejecutar la Aplicación
Sigue estos pasos para poner en marcha el dashboard en tu entorno local.
## 1. Prerrequisitos
   - Python 3.8 o superior.
   - pip y venv (recomendado para entornos virtuales).
## 2. Estructura del Proyecto
Asegúrate de que tu proyecto tenga la siguiente estructura de archivos y carpetas:tu-proyecto/
├── app.py                  # Lógica principal de la aplicación Streamlit
├── utils.py                # Funciones de carga de datos y modelos
├── create_models.py        # Script para entrenar y guardar los modelos (¡ejecutar primero!)
├── requirements.txt        # Dependencias de Python
├── telco_churn.csv         # Dataset (se descarga automáticamente)
└── models/
    ├── rf_pipeline.joblib
    ├── stack_pca.joblib
    └── stack_selected.joblib

## 3. Instalación
Clona o descarga el repositorio y navega a la carpeta del proyecto en tu terminal.
(Recomendado) Crea y activa un entorno virtual:python -m venv venv

### En Windows
```bash
venv\Scripts\activate
```
### En macOS/Linux
```bash
source venv/bin/activate
```
Instala las dependencias:
```bash
pip install -r requirements.txt
```
## 4. Lanza la Aplicación
Una vez que los modelos han sido creados, ejecuta la aplicación con Streamlit:
```bash
streamlit run app.py
```
Se abrirá una nueva pestaña en tu navegador con el dashboard interactivo.

# 🛠️ Funcionamiento del Dashboard

La aplicación está organizada en tres pestañas principales:

# 🔍 Pestaña 1: Explorador y Predicción
Esta es la sección principal para el análisis diario.
- Filtros Globales: En la barra lateral, puedes segmentar a los clientes por tipo de contrato y método de pago. Todos los gráficos de esta pestaña se actualizan según tu selección.
- KPIs del Segmento: Muestra la tasa de fuga y el cargo mensual promedio para el grupo de clientes filtrado.
- Visualizaciones Dinámicas: Contiene secciones para generar gráficos de distribución, box plots, scatter plots y más, permitiéndote elegir las variables a analizar.
- Predicción Individual: Un formulario donde puedes introducir las características de un cliente para obtener una predicción de churn en tiempo real y una recomendación de acción si el riesgo es alto.

# 💼 Pestaña 2: Impacto de Negocio
Esta pestaña traduce los resultados del modelo a un lenguaje financiero y estratégico.
- KPIs de Riesgo: Calcula el porcentaje total de clientes en riesgo y la suma de sus ingresos mensuales (Ingresos en Riesgo).
- Simulador de Retención: Una barra deslizable te permite simular el ahorro mensual que se lograría al retener un cierto porcentaje de los clientes en riesgo.
- Factores Clave de Churn: Muestra un gráfico de importancia de features para identificar visualmente qué características influyen más en la decisión de un cliente de abandonar el servicio.

# 📊 Pestaña 3: Rendimiento de Modelos
Una vista técnica para comparar los diferentes pipelines de Machine Learning.
- Tabla Comparativa: Muestra las métricas de Accuracy, F1-Score y AUC para cada uno de los tres modelos, permitiendo una fácil comparación.
- Matriz de Confusión: Visualiza el rendimiento de un modelo seleccionado, mostrando los verdaderos positivos, falsos positivos, etc.
- Importancia de Features: Un análisis detallado de la influencia de cada variable según el modelo Random Forest.

### 💻 Stack Tecnológico
Lenguaje: Python
Framework Web: Streamlit
Análisis de Datos: Pandas, NumPy
Machine Learning: Scikit-learn
Visualización: Plotly, Matplotlib, Seaborn
Serialización de Modelos: Cloudpickle / Joblib

## 🔐 Configurar la OpenAI API Key (recomendado)

La aplicación puede usar OpenAI para la pestaña de Chat. Nunca dejes tu API key dentro del código. Tienes dos formas seguras de configurarla:

- Opción A — Variable de entorno (recomendada):
    - En Linux / macOS (zsh/bash):
        ```bash
        export OPENAI_API_KEY="sk-...tu_clave_aqui..."
        ```
        Después abre la misma terminal (o exporta en tu .zshrc) y ejecuta:
        ```bash
        streamlit run app.py
        ```

- Opción B — Pegar temporalmente en la sesión (no se guarda en disco):
    - Abre la aplicación y ve a la pestaña ⚙️ Configuración.
    - En el apartado "OpenAI API Key" pega la clave y pulsa "Guardar API Key en sesión".
    - La clave quedará disponible sólo mientras dure la sesión de Streamlit.

Si no hay ninguna clave configurada la app mostrará un mensaje e intentará usar modelos locales si están disponibles (transformers).
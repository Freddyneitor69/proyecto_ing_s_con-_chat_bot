import streamlit as st
import pandas as pd
import joblib
import subprocess
import sys
import json
import os
import tempfile

@st.cache_data
def load_data(filepath):
    """
    Carga el CSV de datos crudos desde una ruta.
    Utiliza el caché de Streamlit para evitar recargar en cada interacción.
    """
    try:
        df = pd.read_csv(filepath)
        # Limpieza básica para las visualizaciones del EDA
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(subset=['TotalCharges'], inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo de datos en '{filepath}'.")
        st.info("Asegúrate de que 'telco_churn.csv' esté en la misma carpeta que la app.")
        return None

@st.cache_resource
def load_model(model_path):
    """
    Carga un pipeline .joblib o .pkl desde la ruta especificada.
    Utiliza el caché de recursos de Streamlit, ideal para objetos pesados como modelos.
    """
    try:
        # Use joblib.load which is the recommended loader for sklearn pipelines
        # and avoids some compatibility issues with cloudpickle when used
        # inside Streamlit's resource cache.
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: No se encontró el modelo en '{model_path}'.")
        st.info(f"Verifica que el archivo exista y la ruta sea correcta.")
        return None
    except Exception as e:
        st.error(f"Ocurrió un error al cargar el modelo en '{model_path}': {e}")
        return None


def run_model_metrics_subprocess(model_path, test_csv_path):
    """Ejecuta la inferencia en un proceso separado usando scripts/predict_helper.py.
    Devuelve un dict con métricas (Accuracy, F1-Score, AUC) o lanza RuntimeError si falla.
    """
    script_path = os.path.join(os.path.dirname(__file__), 'scripts', 'predict_helper.py')
    cmd = [sys.executable, script_path, model_path, test_csv_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Subprocess failed: {proc.stderr.strip()}")
    try:
        out = json.loads(proc.stdout)
        return out.get('metrics', {})
    except Exception as e:
        raise RuntimeError(f"Could not parse subprocess output: {e}\nOutput: {proc.stdout}")


def predict_and_save_subprocess(model_path, input_csv_path):
    """Ejecuta la inferencia en un proceso separado y guarda un CSV con la columna 'prediction'.
    Devuelve la ruta al CSV temporal generado.
    """
    script_path = os.path.join(os.path.dirname(__file__), 'scripts', 'predict_helper.py')
    fd, tmp_path = tempfile.mkstemp(prefix='preds_', suffix='.csv')
    os.close(fd)
    cmd = [sys.executable, script_path, model_path, input_csv_path, tmp_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # Cleanup temp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise RuntimeError(f"Subprocess failed: {proc.stderr.strip()}")
    return tmp_path


def inspect_model_columns_subprocess(model_path):
    """Run the helper in inspect mode to retrieve expected input column names.
    Returns a list of column names or raises RuntimeError with details.
    """
    script_path = os.path.join(os.path.dirname(__file__), 'scripts', 'predict_helper.py')
    cmd = [sys.executable, script_path, '--inspect-columns', model_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Inspect subprocess failed: {proc.stderr.strip()}")
    try:
        out = json.loads(proc.stdout)
        return out.get('expected_columns', None)
    except Exception as e:
        raise RuntimeError(f"Could not parse inspect output: {e}\nOutput: {proc.stdout}")

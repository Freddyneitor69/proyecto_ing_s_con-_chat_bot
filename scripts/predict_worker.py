#!/usr/bin/env python3
"""Worker that actually loads the model and runs predict/predict_proba.
This file contains the original prediction logic. It is executed by
`predict_helper.py` in a subprocess so native crashes only affect the worker.
"""
import sys
import json
import joblib
import pandas as pd
import numpy as np
import traceback
import os
import sys

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def safe_json_print(obj):
    try:
        print(json.dumps(obj))
    except Exception:
        print(json.dumps({"error": "could not serialize output"}))


def main():
    if len(sys.argv) < 2:
        safe_json_print({"error": "usage: predict_worker.py <model_path> [test_csv] [out_predictions_csv]"})
        sys.exit(2)

    inspect_only = False
    if len(sys.argv) >= 2 and sys.argv[1] == '--inspect-columns':
        inspect_only = True
        model_path = sys.argv[2] if len(sys.argv) >= 3 else None
        test_csv = None
        out_predictions = None
    else:
        model_path = sys.argv[1]
        if len(sys.argv) >= 3 and sys.argv[2] == '--inspect-columns':
            inspect_only = True
            test_csv = None
            out_predictions = None
        else:
            test_csv = sys.argv[2] if len(sys.argv) >= 3 else None
            out_predictions = sys.argv[3] if len(sys.argv) > 3 else None

    try:
        # Ensure repository root is on sys.path so custom transformer modules
        # (e.g. models_helpers.py) can be imported during unpickling.
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        model = joblib.load(model_path)
    except Exception as e:
        safe_json_print({"error": f"failed to load model: {e}", "trace": traceback.format_exc()})
        sys.exit(3)

    if inspect_only:
        try:
            from sklearn.compose import ColumnTransformer

            expected = None
            if hasattr(model, 'feature_names_in_'):
                expected = list(model.feature_names_in_)
            else:
                def collect_columns(obj):
                    cols = []
                    try:
                        if isinstance(obj, ColumnTransformer):
                            for name, trans, spec in obj.transformers:
                                if isinstance(spec, (list, tuple)):
                                    cols.extend([c for c in spec if isinstance(c, str)])
                        if hasattr(obj, 'named_steps'):
                            for _n, step in obj.named_steps.items():
                                cols.extend(collect_columns(step))
                        if hasattr(obj, 'steps'):
                            for _n, step in getattr(obj, 'steps'):
                                cols.extend(collect_columns(step))
                    except Exception:
                        pass
                    return cols

                if hasattr(model, 'named_steps'):
                    collected = collect_columns(model)
                    if collected:
                        seen = set()
                        ordered = [x for x in collected if not (x in seen or seen.add(x))]
                        expected = ordered

            safe_json_print({"expected_columns": expected})
            return
        except Exception as e:
            safe_json_print({"error": f"inspection failed: {e}", "trace": traceback.format_exc()})
            sys.exit(8)

    # Read CSV
    try:
        df = pd.read_csv(test_csv)
    except Exception as e:
        safe_json_print({"error": f"failed to read csv '{test_csv}': {e}", "trace": traceback.format_exc()})
        sys.exit(4)

    has_labels = 'Churn' in df.columns
    X = df.drop(columns=['Churn']) if has_labels else df.copy()
    y = df['Churn'].map({'Yes': 1, 'No': 0}) if has_labels else None

    # Determine expected columns
    expected = None
    if hasattr(model, 'feature_names_in_'):
        expected = list(model.feature_names_in_)
    else:
        try:
            from sklearn.compose import ColumnTransformer

            def collect_columns(obj):
                cols = []
                try:
                    if isinstance(obj, ColumnTransformer):
                        for name, trans, spec in obj.transformers:
                            if isinstance(spec, (list, tuple)):
                                cols.extend([c for c in spec if isinstance(c, str)])
                    if hasattr(obj, 'named_steps'):
                        for _n, step in obj.named_steps.items():
                            cols.extend(collect_columns(step))
                    if hasattr(obj, 'steps'):
                        for _n, step in getattr(obj, 'steps'):
                            cols.extend(collect_columns(step))
                except Exception:
                    pass
                return cols

            if hasattr(model, 'named_steps'):
                collected = collect_columns(model)
                if collected:
                    seen = set()
                    ordered = [x for x in collected if not (x in seen or seen.add(x))]
                    expected = ordered
        except Exception:
            expected = None

    if expected is not None:
        missing = [c for c in expected if c not in X.columns]
        if missing:
            for c in missing:
                X[c] = np.nan
        X = X.reindex(columns=expected)

    # Coerce numeric where sensible: try to convert, but avoid using errors='ignore'
    for col in X.columns:
        # Normalize object columns
        if X[col].dtype == 'object':
            try:
                X[col] = X[col].astype(str).str.strip()
            except Exception:
                pass
        # Try numeric conversion; if it fails, keep original
        try:
            converted = pd.to_numeric(X[col])
            # Only assign back if conversion changed dtype to numeric
            if np.issubdtype(converted.dtype, np.number):
                X[col] = converted
        except Exception:
            pass

    # Run prediction
    try:
        y_pred = model.predict(X)
    except Exception as e:
        safe_json_print({"error": f"predict failed: {e}", "trace": traceback.format_exc(), "columns": X.columns.tolist()})
        sys.exit(5)

    y_proba = None
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                y_proba = proba[:, 1]
            elif proba.ndim == 1:
                y_proba = proba
    except Exception:
        y_proba = None

    if out_predictions:
        try:
            out_df = df.copy()
            out_df['prediction'] = [int(x) for x in y_pred]
            if y_proba is not None:
                out_df['prediction_proba'] = [float(x) for x in y_proba]
            out_df.to_csv(out_predictions, index=False)
        except Exception as e:
            safe_json_print({"error": f"failed to write predictions csv: {e}", "trace": traceback.format_exc()})
            sys.exit(6)

    if has_labels:
        try:
            metrics = {
                "Accuracy": float(accuracy_score(y, y_pred)),
                "F1-Score": float(f1_score(y, y_pred)),
                "AUC": float(roc_auc_score(y, y_proba)) if y_proba is not None else None,
            }
            safe_json_print({"metrics": metrics})
        except Exception as e:
            safe_json_print({"error": f"failed to compute metrics: {e}", "trace": traceback.format_exc()})
            sys.exit(7)
    else:
        safe_json_print({"predictions_written": bool(out_predictions), "has_proba": y_proba is not None})


if __name__ == '__main__':
    main()

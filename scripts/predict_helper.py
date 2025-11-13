#!/usr/bin/env python3
"""Helper script to load a model and run predictions over a test CSV.
This script is defensive: it attempts to align incoming columns to the model's
expected features (if available), coerces numeric types where reasonable, and
returns structured JSON on stdout for metric requests or writes a CSV of
predictions when an output path is provided.

Usage:
  python scripts/predict_helper.py <model_path> <test_csv> [out_predictions_csv]
"""
import sys
import json
import joblib
import pandas as pd
import numpy as np
import traceback
import subprocess
import shutil
import os

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def safe_json_print(obj):
    try:
        print(json.dumps(obj))
    except Exception:
        # Last resort: print a compact error
        print(json.dumps({"error": "could not serialize output"}))


def main():
    if len(sys.argv) < 3:
        safe_json_print({"error": "usage: predict_helper.py <model_path> <test_csv> [out_predictions_csv]"})
        sys.exit(2)

    # Support an inspect mode: invocation forms:
    # 1) scripts/predict_helper.py --inspect-columns <model_path>
    # 2) scripts/predict_helper.py <model_path> --inspect-columns
    inspect_only = False
    if len(sys.argv) >= 2 and sys.argv[1] == '--inspect-columns':
        inspect_only = True
        model_path = sys.argv[2] if len(sys.argv) >= 3 else None
        test_csv = None
        out_predictions = None
    else:
        model_path = sys.argv[1]
        # test_csv may be the special flag
        if len(sys.argv) >= 3 and sys.argv[2] == '--inspect-columns':
            inspect_only = True
            test_csv = None
            out_predictions = None
        else:
            test_csv = sys.argv[2] if len(sys.argv) >= 3 else None
            out_predictions = sys.argv[3] if len(sys.argv) > 3 else None

    # Delegate the heavy work to a worker subprocess. This way if a native
    # extension inside sklearn/numpy triggers a segfault it will only crash the
    # worker and we can return a structured error JSON instead of letting the
    # caller see raw stderr/segfault messages.
    python_exe = shutil.which('python') or sys.executable
    worker = os.path.join(os.path.dirname(__file__), 'predict_worker.py')
    if not os.path.exists(worker):
        safe_json_print({"error": "internal: predict_worker.py not found"})
        sys.exit(10)

    # Build command: pass through same arguments used for worker
    cmd = [python_exe, worker, model_path]
    if inspect_only:
        cmd.append('--inspect-columns')
    else:
        if test_csv:
            cmd.append(test_csv)
        if out_predictions:
            cmd.append(out_predictions)

    try:
        # Run worker and capture stdout/stderr
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        safe_json_print({"error": "prediction worker timed out"})
        sys.exit(11)
    except Exception as e:
        safe_json_print({"error": f"failed to start prediction worker: {e}", "trace": traceback.format_exc()})
        sys.exit(12)

    # If worker exited due to signal (segfault) or non-zero exit, surface structured info
    if proc.returncode != 0:
        # If process was terminated by signal, return that info
        rc = proc.returncode
        info = {"error": "prediction worker failed", "returncode": rc, "stderr": proc.stderr}
        # Try to parse JSON from stdout if worker produced any
        try:
            out = json.loads(proc.stdout)
            # prefer structured error from worker
            info.update({"worker_output": out})
        except Exception:
            pass
        safe_json_print(info)
        # exit with non-zero to indicate failure
        sys.exit(5)

    # Worker succeeded: forward its stdout (assumed JSON) to caller
    try:
        # Try to load and re-print to ensure valid JSON
        parsed = json.loads(proc.stdout)
        safe_json_print(parsed)
    except Exception:
        # If worker printed non-JSON, return raw stdout
        safe_json_print({"output": proc.stdout})
    return

    df = None
    if not inspect_only:
        try:
            df = pd.read_csv(test_csv)
        except Exception as e:
            safe_json_print({"error": f"failed to read csv '{test_csv}': {e}", "trace": traceback.format_exc()})
            sys.exit(4)

    if inspect_only:
        # If user only wants inspection, we return the collected expected columns and exit
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

    # If labels are provided, compute metrics. Otherwise just predict and write predictions.
    has_labels = 'Churn' in df.columns

    X = df.drop(columns=['Churn']) if has_labels else df.copy()
    y = df['Churn'].map({'Yes': 1, 'No': 0}) if has_labels else None

    # Determine expected input columns by inspecting the pipeline's preprocessing
    expected = None
    # 1) Most direct: feature_names_in_
    if hasattr(model, 'feature_names_in_'):
        expected = list(model.feature_names_in_)
    else:
        # 2) Walk the pipeline to find ColumnTransformer-specified columns
        try:
            from sklearn.compose import ColumnTransformer

            def collect_columns(obj):
                cols = []
                try:
                    # If obj is a ColumnTransformer, it has .transformers (spec)
                    if isinstance(obj, ColumnTransformer):
                        for name, trans, spec in obj.transformers:
                            # spec can be list of column names or slice or callable
                            if isinstance(spec, (list, tuple)):
                                cols.extend([c for c in spec if isinstance(c, str)])
                    # If obj is a pipeline-like with steps, recurse into steps
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
                    # preserve order but remove duplicates
                    seen = set()
                    ordered = [x for x in collected if not (x in seen or seen.add(x))]
                    expected = ordered
        except Exception:
            expected = None

    if expected is not None:
        # Reindex X to expected columns: missing columns will be filled with NaN
        missing = [c for c in expected if c not in X.columns]
        extra = [c for c in X.columns if c not in expected]
        if missing:
            # fill missing with NaN and continue — pipeline transformers may impute or fail later
            for c in missing:
                X[c] = np.nan
        # keep only expected ordering
        X = X.reindex(columns=expected)

    # Coerce numeric-like columns to numeric where possible to avoid common dtype issues
    for col in X.columns:
        # Skip categorical object columns
        if X[col].dtype == 'object':
            # Attempt to strip whitespace
            X[col] = X[col].astype(str).str.strip()
            continue
        # For other types, attempt numeric coercion
        try:
            X[col] = pd.to_numeric(X[col], errors='ignore')
        except Exception:
            pass

    # Run prediction guardedly and return structured errors on failure
    try:
        y_pred = model.predict(X)
    except Exception as e:
        safe_json_print({"error": f"predict failed: {e}", "trace": traceback.format_exc(), "columns": X.columns.tolist()})
        sys.exit(5)

    y_proba = None
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            # If proba is 2D, take positive class probability if shape matches
            if proba.ndim == 2 and proba.shape[1] >= 2:
                y_proba = proba[:, 1]
            elif proba.ndim == 1:
                y_proba = proba
    except Exception:
        y_proba = None

    # If an output path was provided, write predictions (and probabilities if available)
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
        # Return info that predictions were written
        safe_json_print({"predictions_written": bool(out_predictions), "has_proba": y_proba is not None})


if __name__ == '__main__':
    main()

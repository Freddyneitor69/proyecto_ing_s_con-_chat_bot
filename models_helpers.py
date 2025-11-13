import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TotalChargesTransformer(BaseEstimator, TransformerMixin):
    """Simple transformer to clean and coerce the TotalCharges column.

    Behaviour:
    - If `TotalCharges` exists in the input DataFrame, convert it to numeric
      coercing errors to NaN, strip whitespace, and fill missing values with 0.
    - Returns a DataFrame with the same columns.

    This is intentionally minimal: its goal is to provide a stable,
    importable class matching the typical custom transformer used in
    Telco churn examples so models pickled as `__main__.TotalChargesTransformer`
    can be reloaded and re-saved in this environment.
    """

    def __init__(self, fill_value: float = 0.0):
        self.fill_value = fill_value

    def fit(self, X, y=None):
        # Nothing to fit; return self for sklearn compatibility
        return self

    def transform(self, X):
        # Expect a DataFrame; if not, try to coerce
        if not hasattr(X, 'copy'):
            X = pd.DataFrame(X)
        X = X.copy()
        if 'TotalCharges' in X.columns:
            try:
                # strip whitespace and convert
                X['TotalCharges'] = X['TotalCharges'].astype(str).str.strip()
                X['TotalCharges'] = pd.to_numeric(X['TotalCharges'].replace('', np.nan))
            except Exception:
                # best-effort: coerce with errors='coerce'
                X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
            # fill missing
            X['TotalCharges'] = X['TotalCharges'].fillna(self.fill_value)
        return X

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)

    # Make the transformer play nicely with new sklearn output API
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return np.asarray(input_features)


class SeniorCitizenTransformer(BaseEstimator, TransformerMixin):
    """Normalize SeniorCitizen column to integer 0/1.

    Converts common representations to integers and fills missing with 0.
    """

    def __init__(self, fill_value: int = 0):
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not hasattr(X, 'copy'):
            X = pd.DataFrame(X)
        X = X.copy()
        if 'SeniorCitizen' in X.columns:
            try:
                X['SeniorCitizen'] = pd.to_numeric(X['SeniorCitizen'], errors='coerce')
            except Exception:
                X['SeniorCitizen'] = X['SeniorCitizen'].astype(str).str.extract(r'(\d+)').fillna(0).astype(float)
            X['SeniorCitizen'] = X['SeniorCitizen'].fillna(self.fill_value).astype(int)
        return X


class ServiceNoMerger(BaseEstimator, TransformerMixin):
    """Tidy-up transformer for service columns.

    This transformer performs harmless normalizations such as converting
    'No phone service' to 'No' on common service columns. It is intentionally
    conservative so it doesn't change feature semantics drastically.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not hasattr(X, 'copy'):
            X = pd.DataFrame(X)
        X = X.copy()
        # replace 'No phone service' and similar placeholders with 'No'
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = X[col].astype(str).str.strip().replace({'No phone service': 'No', 'no phone service': 'No'})
                except Exception:
                    pass
        return X

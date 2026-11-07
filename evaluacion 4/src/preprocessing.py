"""Preprocesamiento compartido (compatible con scikit-learn).

Todo transformador aprende sus parámetros en ``fit`` (solo datos de
entrenamiento) y los aplica en ``transform``, evitando fuga de información
cuando se usa dentro de un ``Pipeline`` y validación cruzada.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def _to_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(np.shape(X)[1])])


class DropHighMissing(BaseEstimator, TransformerMixin):
    """Elimina columnas cuyo % de faltantes en train supera ``threshold``."""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = _to_frame(X)
        self.missing_rate_ = X.isna().mean()
        self.keep_ = self.missing_rate_[self.missing_rate_ <= self.threshold].index.tolist()
        self.dropped_ = self.missing_rate_[self.missing_rate_ > self.threshold].index.tolist()
        self.feature_names_in_ = np.array(X.columns)
        return self

    def transform(self, X):
        return _to_frame(X)[self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Elimina una de cada par de variables con |corr| > ``threshold``.

    Se recorre en orden de columnas y se descarta la segunda de cada par.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = _to_frame(X)
        corr = X.corr().abs().to_numpy()
        n = corr.shape[0]
        drop = set()
        for i in range(n):
            if i in drop:
                continue
            for j in range(i + 1, n):
                if j not in drop and corr[i, j] > self.threshold:
                    drop.add(j)
        cols = list(X.columns)
        self.dropped_ = [cols[j] for j in sorted(drop)]
        self.keep_ = [c for k, c in enumerate(cols) if k not in drop]
        return self

    def transform(self, X):
        return _to_frame(X)[self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_)


def signed_log1p(X):
    """log1p con signo: comprime colas largas y tolera negativos."""
    return np.sign(X) * np.log1p(np.abs(X))


def build_preprocessor(
    missing_threshold: float = 0.7,
    add_indicators: bool = True,
    log_transform: bool = True,
    corr_threshold: float | None = 0.95,
    scale: bool = True,
    imputer=None,
) -> Pipeline:
    """Pipeline estándar del equipo.

    1. Elimina columnas con > ``missing_threshold`` de faltantes.
    2. Imputa con mediana (o ``imputer``), opcionalmente con indicadores de NaN.
    3. Transformación log1p con signo (variables muy asimétricas).
    4. Elimina variables constantes.
    5. Filtro de correlación (opcional).
    6. Estandarización (opcional).
    """
    pipe = Pipeline(preprocessor_steps(missing_threshold, add_indicators, log_transform,
                                       corr_threshold, scale, imputer))
    pipe.set_output(transform="pandas")
    return pipe


def preprocessor_steps(
    missing_threshold: float = 0.7,
    add_indicators: bool = True,
    log_transform: bool = True,
    corr_threshold: float | None = 0.95,
    scale: bool = True,
    imputer=None,
) -> list:
    """Mismos pasos que ``build_preprocessor`` como lista ``(nombre, transformador)``.

    Útil para ``imblearn.pipeline.Pipeline``, que no admite pipelines anidados
    (p. ej. preprocesamiento + SMOTE + modelo en una sola cadena).
    """
    if imputer is None:
        imputer = SimpleImputer(strategy="median", add_indicator=add_indicators)
    steps = [("drop_missing", DropHighMissing(missing_threshold)), ("impute", imputer)]
    if log_transform:
        steps.append(
            ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one"))
        )
    steps.append(("variance", VarianceThreshold(0.0)))
    if corr_threshold is not None:
        steps.append(("corr", CorrelationFilter(corr_threshold)))
    if scale:
        steps.append(("scale", StandardScaler()))
    return steps

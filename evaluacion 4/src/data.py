"""Carga del dataset APS Failure at Scania Trucks."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FILES = {
    "train": "aps_failure_training_set.csv",
    "test": "aps_failure_test_set.csv",
}


def _header_line(path: Path) -> int:
    """Índice (0-based) de la línea del encabezado, que empieza con 'class'.

    Los CSV originales traen ~20 líneas de licencia antes del encabezado.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if line.startswith("class,"):
                return i
    raise ValueError(f"No se encontró el encabezado en {path}")


def load_aps(split: str = "train", data_dir: str | Path | None = None):
    """Carga un split ('train' o 'test').

    Returns
    -------
    X : DataFrame (float) con las 170 variables; 'na' -> NaN
    y : Series int (1 = falla APS 'pos', 0 = 'neg')
    """
    if split not in FILES:
        raise ValueError(f"split debe ser uno de {list(FILES)}")
    path = Path(data_dir or DATA_DIR) / FILES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Descarga el dataset (ver data/README.md)."
        )
    df = pd.read_csv(path, skiprows=_header_line(path), na_values="na")
    y = (df.pop("class") == "pos").astype(int).rename("target")
    X = df.astype(float)
    return X, y


def histogram_groups(columns) -> dict[str, list[str]]:
    """Agrupa las variables de histograma (mismo prefijo con bins _000.._009).

    Un prefijo se considera histograma si tiene 10 columnas consecutivas.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for col in columns:
        m = re.match(r"^([a-z]{2})_(\d{3})$", col)
        if m:
            groups[m.group(1)].append(col)
    return {k: sorted(v) for k, v in groups.items() if len(v) == 10}

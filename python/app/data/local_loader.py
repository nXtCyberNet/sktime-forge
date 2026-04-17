from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


def _load_builtin_dataset(dataset_id: str) -> np.ndarray | None:
    key = dataset_id.strip().lower()
    if key == "airline":
        from sktime.datasets import load_airline

        series = load_airline()
        return series.astype(float).to_numpy()
    return None


def _resolve_dataset_path(base_dir: Path, dataset_id: str) -> Path:
    cleaned = dataset_id.strip()
    candidates = [
        base_dir / f"{cleaned}.csv",
        base_dir / cleaned,
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Dataset file not found for dataset_id={dataset_id!r} under {base_dir}"
    )


def build_data_loader(base_dir: str | None = None) -> Callable[[str], np.ndarray]:
    """Return a callable(dataset_id) -> np.ndarray from built-ins + local CSV files."""
    base_path = None
    if base_dir is not None and str(base_dir).strip():
        base_path = Path(base_dir).expanduser().resolve()

    def _load(dataset_id: str) -> np.ndarray:
        builtin = _load_builtin_dataset(dataset_id)
        if builtin is not None:
            if builtin.size < 5:
                raise ValueError(
                    f"Built-in dataset {dataset_id!r} has too few observations ({builtin.size})"
                )
            return builtin

        if base_path is None:
            raise FileNotFoundError(
                f"No local dataset directory configured and dataset_id={dataset_id!r} is not a supported built-in"
            )

        path = _resolve_dataset_path(base_path, dataset_id)
        frame = pd.read_csv(path)

        if "y" in frame.columns:
            series = frame["y"]
        else:
            numeric_cols = [
                col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])
            ]
            if not numeric_cols:
                raise ValueError(
                    f"Dataset {path} does not contain numeric columns or a 'y' column"
                )
            series = frame[numeric_cols[-1]]

        values = series.astype(float).to_numpy()
        if values.size < 5:
            raise ValueError(
                f"Dataset {path} has too few observations ({values.size}); need at least 5"
            )
        return values

    return _load


def build_csv_data_loader(base_dir: str):
    """Backward-compatible alias for older imports."""
    return build_data_loader(base_dir)

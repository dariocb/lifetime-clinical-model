"""I/O helpers for the canonical synthetic clinical dataset."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from .synthetic import SyntheticConfig, SyntheticDataset

_DATE_COLUMNS = {
    "patients": ["date_of_birth", "ehr_recorded_at", "study_start_date", "study_end_date"],
    "events": ["event_date"],
    "pre_study_events": ["event_date"],
    "measurements": ["measurement_date"],
    "dynamic_covariates": ["recorded_at"],
}


def _read_table(root: Path, name: str) -> pd.DataFrame:
    path = root / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    for column in _DATE_COLUMNS.get(name, []):
        if column in df:
            df[column] = pd.to_datetime(df[column])
    return df


def load_dataset(directory) -> SyntheticDataset:
    """Load the tidy CSV dataset produced by ``SyntheticDataset.save``."""
    root = Path(directory)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    config = SyntheticConfig(**metadata["config"])
    return SyntheticDataset(
        patients=_read_table(root, "patients"),
        events=_read_table(root, "events"),
        pre_study_events=_read_table(root, "pre_study_events"),
        measurements=_read_table(root, "measurements"),
        config=config,
        dynamic_covariates=_read_table(root, "dynamic_covariates"),
        embedding_metadata=metadata.get("embedding", {}),
    )


def load_splits(directory) -> dict[str, list[str]]:
    path = Path(directory) / "splits.json"
    return json.loads(path.read_text(encoding="utf-8"))

"""Load TRM observations from local files."""
from pathlib import Path
import pandas as pd
from .validation import validate_trm


def load_trm(path: str | Path, date_column: str = "date", value_column: str = "trm") -> pd.DataFrame:
    """Read CSV or Parquet data and return a validated date-indexed frame."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    if file_path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(file_path)
    elif file_path.suffix.lower() in {".csv", ".txt"}:
        raw = pd.read_csv(file_path)
    else:
        raise ValueError("supported input formats are CSV and Parquet")
    return validate_trm(raw, date_column=date_column, value_column=value_column)

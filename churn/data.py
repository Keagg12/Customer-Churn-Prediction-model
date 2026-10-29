"""Load a tabular file and describe it. Makes no assumption about column names."""
import csv
import io
from pathlib import Path

import pandas as pd


class DataLoadError(Exception):
    """Raised with a plain-language message when a file cannot be read."""


_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
_EXCEL = (".xlsx", ".xlsm", ".xls")


def _read_bytes(source) -> bytes:
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    if hasattr(source, "seek"):
        source.seek(0)
    raw = source.read()
    return raw.encode("utf-8") if isinstance(raw, str) else raw


def load_table(source, filename=None, sheet=0) -> pd.DataFrame:
    """Read CSV/TSV/text (encoding and delimiter detected) or Excel into a DataFrame."""
    name = (filename or (str(source) if isinstance(source, (str, Path)) else getattr(source, "name", ""))).lower()
    try:
        if name.endswith(_EXCEL):
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_excel(source, sheet_name=sheet)
        else:
            raw = _read_bytes(source)
            if not raw.strip():
                raise DataLoadError("The file is empty.")
            text = None
            for enc in _ENCODINGS:
                try:
                    text = raw.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            if text is None:
                raise DataLoadError("Could not decode the file as text.")
            try:
                delim = csv.Sniffer().sniff(text[:20000], delimiters=",;\t|").delimiter
            except csv.Error:
                delim = ","
            df = pd.read_csv(io.StringIO(text), sep=delim)
    except DataLoadError:
        raise
    except Exception as exc:  # surface any parser problem in plain language
        raise DataLoadError(f"Could not read the file: {exc}") from exc
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise DataLoadError("The file was read but contains no rows or no columns.")
    return df


def overview(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "fully_empty_columns": [c for c in df.columns if df[c].isna().all()],
    }


def profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        nn = s.dropna()
        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_pct": round(100 * float(s.isna().mean()), 2),
            "unique": int(nn.nunique()),
            "examples": ", ".join(str(v) for v in nn.unique()[:3]),
        })
    return pd.DataFrame(rows)

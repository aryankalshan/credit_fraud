"""One dataset, disjoint splits, and training-only preprocessing."""
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]


def load_data(path, download=False):
    path = Path(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Missing {path}. Run with --download or supply --data.")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".part")
        print("Downloading the standard credit-card dataset...", flush=True)
        urlretrieve(DATA_URL, temporary)
        temporary.replace(path)
    frame = pd.read_csv(path)
    if set(frame.columns) != set(FEATURES + ["Class"]):
        raise ValueError("Expected Time, V1...V28, Amount, Class, with no extra columns.")
    frame = frame[FEATURES + ["Class"]].apply(pd.to_numeric, errors="raise")
    if not frame.Class.isin([0, 1]).all() or frame.Class.nunique() != 2:
        raise ValueError("Class must contain both 0 (normal) and 1 (fraud), without missing labels.")
    if np.isinf(frame.to_numpy()).any():
        raise ValueError("Infinite values are not supported.")
    if frame.drop_duplicates().duplicated(subset=FEATURES).any():
        raise ValueError("Identical features have conflicting labels; resolve these records first.")
    original = len(frame)
    frame = frame.drop_duplicates().copy()  # Prevent identical records crossing splits.
    frame.index.name = "row_id"  # Preserve the original CSV row throughout every feature step.
    frame["Class"] = frame.Class.astype("int8")
    summary = {"original_rows": original, "duplicates_removed": original - len(frame),
               "rows": len(frame), "frauds": int(frame.Class.sum()),
               "fraud_ratio": float(frame.Class.mean()),
               "missing_values": int(frame.isna().sum().sum())}
    return frame, summary


def split_data(frame, method="temporal", seed=42):
    if method == "temporal":
        if frame.Time.isna().any():
            raise ValueError("Temporal splitting requires complete Time values.")
        ordered = frame.sort_values("Time", kind="stable")
        times = ordered.Time.to_numpy()
        # Keep equal timestamps on the same side of each chronological boundary.
        first = np.searchsorted(times, times[int(0.6 * len(times))])
        second = np.searchsorted(times, times[int(0.8 * len(times))])
        parts = [ordered.iloc[:first], ordered.iloc[first:second], ordered.iloc[second:]]
    elif method == "stratified":
        development, test = train_test_split(frame, test_size=0.2, stratify=frame.Class,
                                             random_state=seed)
        train, valid = train_test_split(development, test_size=0.25,
                                       stratify=development.Class, random_state=seed)
        parts = [train, valid, test]
    else:
        raise ValueError("Split must be temporal or stratified.")
    for part in parts:
        if part.Class.value_counts().reindex([0, 1], fill_value=0).min() < 10:
            raise ValueError("Each split needs at least 10 examples of each class for evaluation.")
    assert len(set().union(*(set(p.index) for p in parts))) == sum(map(len, parts))
    return dict(zip(["train", "validation", "test"], parts))


def fit_preprocessor(train):
    if train[FEATURES].isna().all().any():
        raise ValueError("A training feature is entirely missing.")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    raw = imputer.fit_transform(train[FEATURES])
    scaler = StandardScaler().fit(raw)
    return {"imputer": imputer, "scaler": scaler, "features": FEATURES}


def transform(frame, preprocessor):
    raw = pd.DataFrame(preprocessor["imputer"].transform(frame[FEATURES]),
                       index=frame.index, columns=FEATURES).astype("float32")
    scaled = preprocessor["scaler"].transform(raw.to_numpy()).astype("float32")
    if not np.isfinite(scaled).all():
        raise ValueError("Non-finite model inputs after preprocessing.")
    return raw, scaled

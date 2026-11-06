"""shared helpers: gold table readers, model registry, psi / csi"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from utils import config as C

ID_COLS = ["loan_id", "Customer_ID", "snapshot_date"]
EPS = 1e-4


# gold readers
def _partitions(directory, upto=None, only=None):
    """partition paths of a gold table up to a date"""
    out = []
    for p in sorted(glob.glob(directory + "*")):
        m = re.search(r"(\d{4}_\d{2}_\d{2})", os.path.basename(p))
        if not m:
            continue
        d = m.group(1).replace("_", "-")
        if (upto and d > upto) or (only and d != only):
            continue
        out.append(p)
    return out


def read_gold(directory, upto=None, only=None):
    paths = _partitions(directory, upto, only)
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in paths]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def feature_columns(df):
    return [c for c in df.columns if c not in ID_COLS]


def load_training_frame(snapshot_date):
    """labelled loans known at snapshot_date (a loan applied in month M is labelled in M+6)"""
    labels = read_gold(C.GOLD_LABEL, upto=snapshot_date)
    feats = read_gold(C.GOLD_FEATURE, upto=C.month_add(snapshot_date, -C.MOB))
    labels = labels[["loan_id", "label"]]
    df = feats.merge(labels, on="loan_id", how="inner")
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.strftime("%Y-%m-%d")
    return df


# model registry
REGISTRY = C.MODEL_BANK + "registry.json"


def read_registry():
    if os.path.exists(REGISTRY):
        with open(REGISTRY) as f:
            return json.load(f)
    return {"models": [], "production": [], "review_log": [], "outage_events": []}


def write_registry(reg):
    os.makedirs(C.MODEL_BANK, exist_ok=True)
    with open(REGISTRY, "w") as f:
        json.dump(reg, f, indent=2, default=str)


def production_version(snapshot_date, reg=None, exclude=None):
    """model version in production on snapshot_date"""
    reg = reg or read_registry()
    live = [p for p in reg["production"] if p["effective_date"] <= snapshot_date and p["version"] != exclude]
    return max(live, key=lambda p: (p["effective_date"], p["promoted_at_order"]))["version"] if live else None


def model_entry(version, reg=None):
    reg = reg or read_registry()
    return next((m for m in reg["models"] if m["version"] == version), None)


def load_artefact(version):
    import pickle
    # registry keeps only the file name
    with open(os.path.join(C.MODEL_BANK, os.path.basename(model_entry(version)["path"])), "rb") as f:
        return pickle.load(f)


# feature groups and coverage
def group_columns(features, group):
    test = C.FEATURE_GROUPS[group]
    return [f for f in features if test(f)]


def dependent_groups(features):
    """feature groups the model uses as inputs"""
    return [g for g in C.FEATURE_GROUPS if group_columns(features, g)]


def batch_coverage(snapshot_date):
    """share of applicants with data per feature group in this month's batch"""
    paths = _partitions(C.GOLD_FEATURE, only=snapshot_date)
    if not paths:
        return None
    cols = list(C.GROUP_COVERAGE_COLUMN.values())
    df = pd.read_parquet(paths[0], columns=cols)
    if df.empty:
        return None
    return {g: float(df[c].mean()) for g, c in C.GROUP_COVERAGE_COLUMN.items()}


def failing_groups(coverage):
    return [g for g, v in (coverage or {}).items() if v < C.MIN_GROUP_COVERAGE]


def outage_events(upto):
    """outage events per feature group: starts when coverage < MIN_GROUP_COVERAGE, ends when it is back at >= MIN_GROUP_COVERAGE"""
    events, open_ev = [], {}
    d = C.START_DATE
    while d <= upto:
        cov = batch_coverage(d)
        if cov is not None:
            for g, v in cov.items():
                if v < C.MIN_GROUP_COVERAGE:
                    if g not in open_ev:
                        open_ev[g] = {"group": g, "start": d, "end": None, "min_coverage": v}
                        events.append(open_ev[g])
                    open_ev[g]["min_coverage"] = min(open_ev[g]["min_coverage"], v)
                elif g in open_ev:
                    open_ev[g]["end"] = d
                    del open_ev[g]
        d = C.month_add(d, 1)
    return events


def months_between(d1, d2):
    a, b = pd.Timestamp(d1), pd.Timestamp(d2)
    return (b.year - a.year) * 12 + (b.month - a.month)


def last_attempt(reg, before, inclusive=False):
    """latest training attempt, promoted or not"""
    att = [m for m in reg["models"] if (m["train_date"] <= before if inclusive else m["train_date"] < before)]
    return max(att, key=lambda m: m["train_date"]) if att else None


def input_status(version, snapshot_date, reg=None):
    """normal or degraded_review (champion uses a feature group that is in outage in this batch)"""
    cov = batch_coverage(snapshot_date)
    art_features = model_entry(version, reg)["features"]
    bad = [g for g in failing_groups(cov) if g in dependent_groups(art_features)]
    return ("degraded_review" if bad else "normal"), bad


def simulate_outage(X, groups):
    """set the groups' columns to how the gold layer shows a missing feed (0)"""
    X = X.copy()
    for g in groups:
        for c in group_columns(list(X.columns), g):
            X[c] = 0
    return X


# psi / csi
def make_bin_spec(values, max_categories=10, n_bins=10):
    """bins from the training data only"""
    v = pd.Series(values).astype(float)
    uniq = np.sort(v.unique())
    if len(uniq) <= max_categories:
        spec = {"type": "cat", "values": [float(u) for u in uniq]}
    else:
        edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1]))
        spec = {"type": "num", "edges": [float(e) for e in edges]}
    spec["props"] = bin_proportions(v, spec)
    return spec


def bin_proportions(values, spec):
    v = pd.Series(values).astype(float).to_numpy()
    if spec["type"] == "cat":
        known = spec["values"]
        idx = np.array([known.index(x) if x in known else len(known) for x in v])
        n = len(known) + 1                      # last bin for unseen values
    else:
        idx = np.searchsorted(spec["edges"], v, side="right")
        n = len(spec["edges"]) + 1
    return (np.bincount(idx, minlength=n) / max(len(v), 1)).tolist()


def psi(expected, actual):
    e = np.clip(np.asarray(expected, float), EPS, None)
    a = np.clip(np.asarray(actual, float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_vs_spec(values, spec):
    return psi(spec["props"], bin_proportions(values, spec))


def psi_band(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return "RED" if x >= C.PSI_RED else "AMBER" if x >= C.PSI_AMBER else "GREEN"

"""monitor the production model for one month, saved as a gold table

stability (now): score psi and feature csi of this month's applications, alerts use psi vs the oot reference
performance (6 months later): auc, gini, ks and calibration of the cohort whose labels arrive this month

usage: python utils/model_monitor.py --snapshotdate 2024-10-01
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

from utils import config as C
from utils import model_common as M

NAN = float("nan")


def performance(snapshot_date):
    """metrics of the cohort whose labels arrive this month"""
    cohort = C.month_add(snapshot_date, -C.MOB)
    out = {"perf_cohort": cohort, "perf_n": 0, "perf_model_version": None, "perf_bad_rate": NAN, "perf_mean_pred": NAN,
           "auc": NAN, "gini": NAN, "ks": NAN, "brier": NAN, "calibration_gap": NAN, "precision": NAN, "recall": NAN,
           "f1": NAN, "baseline_auc": NAN, "auc_drop": NAN}
    preds = M.read_gold(C.GOLD_PREDICTIONS, only=cohort)
    labels = M.read_gold(C.GOLD_LABEL, only=snapshot_date)
    if preds.empty or labels.empty:
        return out
    df = preds.merge(labels[["loan_id", "label"]], on="loan_id", how="inner")
    if df["label"].nunique() < 2:
        return out
    v = df["model_version"].iloc[0]                      # model that scored the cohort
    entry = M.model_entry(v)
    y, s, pred = df["label"].to_numpy(), df["score"].to_numpy(), df["prediction"].to_numpy()
    auc = roc_auc_score(y, s)
    out.update({"perf_n": int(len(df)), "perf_model_version": v, "perf_bad_rate": float(y.mean()),
                "perf_mean_pred": float(s.mean()), "auc": float(auc), "gini": float(2 * auc - 1),
                "ks": float(ks_2samp(s[y == 1], s[y == 0]).statistic), "brier": float(brier_score_loss(y, s)),
                "calibration_gap": float(abs(s.mean() - y.mean())),
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)), "f1": float(f1_score(y, pred, zero_division=0)),
                "baseline_auc": float(entry["oot_auc"]), "auc_drop": float(entry["oot_auc"] - auc)})
    return out


def stability(snapshot_date, art):
    empty = {"n_scored": 0, "mean_score": NAN, "predicted_default_rate": NAN, "score_psi": NAN, "score_psi_train_ref": NAN,
             "max_csi_top20": NAN, "n_csi_red_top20": 0, "n_csi_amber_top20": 0, "top_drift_feature": None,
             "has_clickstream_rate": NAN}
    X = M.read_gold(C.GOLD_FEATURE, only=snapshot_date)
    preds = M.read_gold(C.GOLD_PREDICTIONS, only=snapshot_date)
    if X.empty or preds.empty:
        return empty, pd.DataFrame()
    csi = pd.DataFrame({"feature": art["features"],
                        "csi": [M.psi_vs_spec(X[f], art["ref_features_oot"][f]) for f in art["features"]],
                        "csi_train_ref": [M.psi_vs_spec(X[f], art["ref_features_train"][f]) for f in art["features"]]})
    csi["is_top20"] = csi["feature"].isin(art["top_features"])
    csi["importance"] = csi["feature"].map(art["importance"])
    top = csi[csi["is_top20"]]
    worst = top.sort_values("csi", ascending=False).iloc[0]
    res = {"n_scored": int(len(preds)), "mean_score": float(preds["score"].mean()),
           "predicted_default_rate": float(preds["prediction"].mean()),
           "score_psi": M.psi_vs_spec(preds["score"], art["ref_score_oot"]),
           "score_psi_train_ref": M.psi_vs_spec(preds["score"], art["ref_score_train"]),
           "max_csi_top20": float(worst["csi"]), "n_csi_red_top20": int((top["csi"] >= C.PSI_RED).sum()),
           "n_csi_amber_top20": int(((top["csi"] >= C.PSI_AMBER) & (top["csi"] < C.PSI_RED)).sum()),
           "top_drift_feature": worst["feature"],
           "has_clickstream_rate": float(X["has_clickstream"].mean()) if "has_clickstream" in X else NAN}
    return res, csi


def auc_band(drop):
    if drop is None or np.isnan(drop):
        return "n/a"
    return "RED" if drop >= C.AUC_DROP_RED else "AMBER" if drop >= C.AUC_DROP_AMBER else "GREEN"


def calibration_band(gap):
    if gap is None or np.isnan(gap):
        return "n/a"
    return "RED" if gap >= C.CALIBRATION_RED else "AMBER" if gap >= C.CALIBRATION_AMBER else "GREEN"


def monitor(snapshot_date):
    version = M.production_version(snapshot_date)
    if version is None:
        print("no model in production on", snapshot_date, "- nothing to monitor")
        return None
    art = M.load_artefact(version)
    stab, csi = stability(snapshot_date, art)
    perf = performance(snapshot_date)
    in_status, in_bad = M.input_status(version, snapshot_date) if stab["n_scored"] else ("n/a", [])

    # stability, performance and calibration status
    s_band = M.psi_band(stab["score_psi"])
    if s_band != "n/a" and stab["n_csi_red_top20"] >= C.CSI_RED_FEATURES:
        s_band = "RED"
    p_band = auc_band(perf["auc_drop"])
    c_band = calibration_band(perf["calibration_gap"])
    bands = [b for b in (s_band, p_band, c_band) if b != "n/a"]
    overall = "RED" if "RED" in bands else "AMBER" if "AMBER" in bands else "GREEN" if bands else "n/a"

    row = {"snapshot_date": snapshot_date, "model_version": version, "model_name": art["model_name"],
           "input_status": in_status, "degraded_features": ",".join(in_bad),
           **stab, **perf, "stability_status": s_band, "performance_status": p_band, "calibration_status": c_band,
           "overall_status": overall}
    os.makedirs(C.GOLD_MONITORING, exist_ok=True)
    pd.DataFrame([row]).to_parquet(f"{C.GOLD_MONITORING}model_monitoring_{C.tag(snapshot_date)}.parquet", index=False)
    if len(csi):
        os.makedirs(C.GOLD_CSI, exist_ok=True)
        csi.assign(snapshot_date=snapshot_date, model_version=version).to_parquet(
            f"{C.GOLD_CSI}model_monitoring_csi_{C.tag(snapshot_date)}.parquet", index=False)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshotdate", required=True)
    a = p.parse_args()
    os.chdir(C.BASE_DIR)
    monitor(a.snapshotdate)

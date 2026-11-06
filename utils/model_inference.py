"""score one month of applications with the production model, saved as a gold table

usage: python utils/model_inference.py --snapshotdate 2024-05-01
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config as C
from utils import model_common as M


def infer(snapshot_date):
    version = M.production_version(snapshot_date)
    if version is None:
        reg = M.read_registry()
        pending = [r for r in reg.get("review_log", []) if r["kind"] == "initial_not_live"]
        print("no model in production on", snapshot_date, "- nothing to score"
              + (" (initial candidate did not meet the floor: not live, manual review pending)" if pending else ""))
        return None
    X = M.read_gold(C.GOLD_FEATURE, only=snapshot_date)
    if X.empty:
        print("no applications on", snapshot_date)
        return None
    art = M.load_artefact(version)
    score = art["model"].predict_proba(X[art["features"]])[:, 1]
    out = X[["loan_id", "Customer_ID", "snapshot_date"]].copy()
    out["model_version"], out["model_name"] = version, art["model_name"]
    out["score"] = score.astype("float64")
    out["prediction"] = (score >= art["threshold"]).astype(int)
    # flag when inputs are degraded and no replacement is qualified
    status, bad = M.input_status(version, snapshot_date)
    out["input_status"] = status
    out["degraded_features"] = ",".join(bad)
    os.makedirs(C.GOLD_PREDICTIONS, exist_ok=True)
    path = f"{C.GOLD_PREDICTIONS}model_predictions_{C.tag(snapshot_date)}.parquet"
    out.to_parquet(path, index=False)
    print(f"scored {len(out)} loans with {version} ({art['model_name']}) -> {path}; "
          f"predicted default rate {out['prediction'].mean():.3f}; input_status={status}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshotdate", required=True)
    a = p.parse_args()
    os.chdir(C.BASE_DIR)
    infer(a.snapshotdate)

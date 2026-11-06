"""train candidate models, evaluate, and decide whether the best one can go live

leakage: matured labels only, train < validation < oot in time (asserted), scaler and model fit on training rows only
validation picks the algorithm and parameters, the choice is refit on train + validation,
oot is for the final evaluation, floor checks and the champion comparison (so oot auc is slightly optimistic)

usage: python utils/model_train.py --snapshotdate 2024-04-01 --mode pre_inference|post_monitoring [--effective DATE]
"""
import argparse
import os
import pickle
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from utils import config as C
from utils import model_common as M
from utils import model_governance as G

warnings.filterwarnings("ignore")


def metrics(y, score, threshold):
    pred = (score >= threshold).astype(int)
    auc = roc_auc_score(y, score)
    return {
        "n": int(len(y)), "bad_rate": float(np.mean(y)), "mean_pred": float(np.mean(score)),
        "calibration_gap": float(abs(np.mean(score) - np.mean(y))), "brier": float(brier_score_loss(y, score)),
        "auc": float(auc), "gini": float(2 * auc - 1),
        "ks": float(ks_2samp(score[y == 1], score[y == 0]).statistic),
        "avg_precision": float(average_precision_score(y, score)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def best_f1_threshold(y, score):
    grid = np.unique(np.quantile(score, np.linspace(0.05, 0.95, 91)))
    return float(max(grid, key=lambda t: f1_score(y, (score >= t).astype(int), zero_division=0)))


def candidates():
    """candidate models with small fixed grids"""
    out = []
    for c in (0.01, 0.1, 1.0):
        out.append(("logistic_regression", {"C": c}, Pipeline([
            ("scale", StandardScaler()), ("clf", LogisticRegression(C=c, max_iter=3000))])))
    for depth, leaf in ((6, 30), (8, 60), (12, 30)):
        out.append(("random_forest", {"max_depth": depth, "min_samples_leaf": leaf}, Pipeline([
            ("clf", RandomForestClassifier(n_estimators=200, max_depth=depth, min_samples_leaf=leaf,
                                           max_features="sqrt", random_state=C.RANDOM_STATE, n_jobs=2))])))
    for n, depth, lr in ((200, 2, 0.05), (200, 3, 0.05), (300, 3, 0.03), (200, 4, 0.05), (150, 3, 0.10), (300, 2, 0.05)):
        out.append(("xgboost", {"n_estimators": n, "max_depth": depth, "learning_rate": lr}, Pipeline([
            ("clf", XGBClassifier(n_estimators=n, max_depth=depth, learning_rate=lr, subsample=0.8,
                                  colsample_bytree=0.8, min_child_weight=10, eval_metric="auc",
                                  random_state=C.RANDOM_STATE, n_jobs=2, tree_method="hist"))])))
    return out


def importances(model, feats):
    est = model.named_steps["clf"]
    imp = np.abs(est.coef_[0]) if hasattr(est, "coef_") else est.feature_importances_
    return dict(sorted(zip(feats, map(float, imp)), key=lambda kv: -kv[1]))


def temporal_split(df, snapshot_date):
    cohorts = sorted(df["snapshot_date"].unique())
    oot_c = cohorts[-C.OOT_COHORTS:]
    val_c = cohorts[-(C.OOT_COHORTS + C.VALIDATION_COHORTS):-C.OOT_COHORTS]
    train_c = cohorts[:-(C.OOT_COHORTS + C.VALIDATION_COHORTS)]
    # check leakage
    assert cohorts[-1] <= C.month_add(snapshot_date, -C.MOB), "a cohort without a matured label is in the data"
    assert train_c and train_c[-1] < val_c[0] < val_c[-1] < oot_c[0], "windows must be ordered train < val < oot"
    part = {k: df[df["snapshot_date"].isin(v)] for k, v in (("train", train_c), ("val", val_c), ("oot", oot_c))}
    return part, {"train": train_c, "val": val_c, "oot": oot_c}, cohorts


def score_of(model, X):
    return model.predict_proba(X)[:, 1]


def train(snapshot_date, effective_date, trigger, why=""):
    df = M.load_training_frame(snapshot_date)
    part, wins, cohorts = temporal_split(df, snapshot_date)
    version = "v" + C.tag(snapshot_date)

    # drop feature groups in outage
    cov = M.batch_coverage(snapshot_date) or {}
    removed = M.failing_groups(cov)
    all_feats = M.feature_columns(df.drop(columns=["label"]))
    feats = [f for f in all_feats if not any(f in M.group_columns(all_feats, g) for g in removed)]
    X = {k: p[feats] for k, p in part.items()}
    y = {k: p["label"].to_numpy() for k, p in part.items()}
    print(f"cohorts {cohorts[0]}..{cohorts[-1]} ({len(cohorts)}) | train {len(X['train'])} val {len(X['val'])} "
          f"oot {len(X['oot'])} | features {len(feats)} (removed groups: {removed or 'none'}, coverage {cov})")

    # fit candidates on train, rank on validation
    board = []
    for family, params, pipe in candidates():
        m = clone(pipe).fit(X["train"], y["train"])
        board.append({"family": family, "params": params,
                      "train_auc": float(roc_auc_score(y["train"], score_of(m, X["train"]))),
                      "val_auc": float(roc_auc_score(y["val"], score_of(m, X["val"]))), "_pipe": pipe, "_fit": m})
    board.sort(key=lambda r: r["val_auc"], reverse=True)
    best = board[0]
    thr = best_f1_threshold(y["val"], score_of(best["_fit"], X["val"]))
    for r in board[:6]:
        print(f"   {r['family']:20s} {str(r['params']):62s} train {r['train_auc']:.3f} val {r['val_auc']:.3f}")

    # refit the best on train + validation
    X_tv = pd.concat([X["train"], X["val"]])
    y_tv = np.concatenate([y["train"], y["val"]])
    model = clone(best["_pipe"]).fit(X_tv, y_tv)

    # evaluate on oot
    s_oot = score_of(model, X["oot"])
    oot = metrics(y["oot"], s_oot, thr)
    imp = importances(model, feats)
    top = list(imp)[:20]

    # psi / csi references
    ref_score_train = M.make_bin_spec(score_of(model, X_tv))
    ref_score_oot = M.make_bin_spec(s_oot)
    ref_feat_train = {f: M.make_bin_spec(X_tv[f]) for f in feats}
    ref_feat_oot = {f: M.make_bin_spec(X["oot"][f]) for f in feats}

    floor_ok = oot["auc"] >= C.MIN_OOT_AUC and oot["calibration_gap"] <= C.MAX_CALIBRATION_GAP
    floor_txt = (f"OOT AUC {oot['auc']:.4f} (floor {C.MIN_OOT_AUC}), calibration gap {oot['calibration_gap']:.4f} "
                 f"(max {C.MAX_CALIBRATION_GAP})")

    # champion vs challenger
    reg = M.read_registry()
    champ_v = M.production_version(effective_date, reg, exclude=version)
    champ_normal = champ_stress = None
    basis = None
    champ_saw_oot = None
    if champ_v is None:
        promote = floor_ok
        why_dec = ("first model: floor met - " if promote else "first model: floor NOT met - ") + floor_txt
        gain = None
    else:
        champ = M.load_artefact(champ_v)
        Xc = part["oot"][champ["features"]]
        champ_normal = float(roc_auc_score(y["oot"], score_of(champ["model"], Xc)))
        stress_groups = [g for g in M.failing_groups(cov) if g in M.dependent_groups(champ["features"])]
        if stress_groups:
            champ_stress = float(roc_auc_score(y["oot"], score_of(champ["model"], M.simulate_outage(Xc, stress_groups))))
        basis = "stress" if (trigger == "feature_outage" and champ_stress is not None) else "normal"
        ref_auc = champ_stress if basis == "stress" else champ_normal
        gain = oot["auc"] - ref_auc
        champ_saw_oot = bool(max(champ["cohorts"]) >= min(wins["oot"]))
        promote = floor_ok and gain >= C.MIN_AUC_GAIN
        why_dec = (f"{'promote' if promote else 'keep champion'}: candidate OOT AUC {oot['auc']:.4f} vs champion "
                   f"{basis} OOT AUC {ref_auc:.4f} (gain {gain:+.4f}, needs {C.MIN_AUC_GAIN}); {floor_txt}"
                   f"{'' if floor_ok else ' - FLOOR NOT MET'}")

    artefact = {
        "version": version, "model_name": best["family"], "model": model, "features": feats, "threshold": thr,
        "train_date": snapshot_date, "label_def": f"{C.DPD}dpd_{C.MOB}mob", "cohorts": cohorts, "windows": wins,
        "metrics_oot": oot, "importance": imp, "top_features": top, "removed_groups": removed, "coverage": cov,
        "ref_score_train": ref_score_train, "ref_score_oot": ref_score_oot,
        "ref_features_train": ref_feat_train, "ref_features_oot": ref_feat_oot,
    }
    os.makedirs(C.MODEL_BANK, exist_ok=True)
    path = f"{C.MODEL_BANK}loan_default_{best['family']}_{C.tag(snapshot_date)}.pkl"
    with open(path, "wb") as f:
        pickle.dump(artefact, f)

    entry = {
        "version": version, "model_name": best["family"], "best_params": best["params"], "train_date": snapshot_date,
        "path": os.path.basename(path), "trigger": trigger, "trigger_reason": why, "effective_date": effective_date,
        "n_cohorts": len(cohorts), "windows": wins, "n_train": int(len(X["train"])), "n_val": int(len(X["val"])),
        "n_oot": int(len(X["oot"])), "features": feats, "n_features": len(feats), "removed_groups": removed,
        "coverage_at_training": cov, "dependent_groups": M.dependent_groups(feats),
        "selected_on": "validation_auc", "val_auc": best["val_auc"], "oot_auc": oot["auc"], "oot_gini": oot["gini"],
        "oot_calibration_gap": oot["calibration_gap"], "oot_brier": oot["brier"], "floor_met": bool(floor_ok),
        "champion": champ_v, "champion_oot_auc_normal": champ_normal, "champion_oot_auc_stress_test": champ_stress,
        "comparison_basis": basis, "auc_gain": gain, "champion_trained_on_oot_cohorts": champ_saw_oot,
        "promoted": bool(promote), "decision": why_dec,
        "leaderboard": [{k: v for k, v in r.items() if not k.startswith("_")} for r in board],
    }
    reg["models"] = [m for m in reg["models"] if m["version"] != version] + [entry]
    reg["production"] = [p for p in reg["production"] if p["version"] != version]
    reg["review_log"] = [r for r in reg.get("review_log", []) if r["version"] != version]
    if promote:
        reg["production"].append({"version": version, "effective_date": effective_date,
                                  "promoted_at_order": len(reg["production"])})
    elif champ_v is None:
        reg["review_log"].append({"date": snapshot_date, "version": version, "kind": "initial_not_live",
                                  "needs_review": True, "reason": "no candidate met the floor: not going live, manual review"})
    elif trigger == "feature_outage":
        reg["review_log"].append({"date": snapshot_date, "version": version, "kind": "no_qualified_replacement",
                                  "needs_review": True,
                                  "reason": "champion inputs degraded and no candidate qualified: scores flagged degraded_review"})
    else:
        reg["review_log"].append({"date": snapshot_date, "version": version, "kind": "champion_retained",
                                  "needs_review": False, "reason": why_dec})
    reg["outage_events"] = M.outage_events(snapshot_date)
    M.write_registry(reg)
    print(f"selected {best['family']} ({version}) trigger={trigger}; promoted={promote} - {why_dec}; effective {effective_date}")
    return entry


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snapshotdate", required=True)
    p.add_argument("--mode", required=True, choices=["pre_inference", "post_monitoring"])
    p.add_argument("--effective", default=None, help="date the model goes live (default: this month, or next month for post_monitoring)")
    a = p.parse_args()
    os.chdir(C.BASE_DIR)
    if a.mode == "pre_inference":
        trigger, why = G.pre_inference_trigger(a.snapshotdate)
        effective = a.effective or a.snapshotdate
    else:
        trigger, why = G.post_monitoring_trigger(a.snapshotdate)
        effective = a.effective or C.month_add(a.snapshotdate, 1)
    if trigger is None:
        print("no training triggered:", why)
        sys.exit(0)
    train(a.snapshotdate, effective, trigger, why)

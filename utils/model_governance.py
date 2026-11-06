"""governance rules: when to train a model, and whether it can go live

assume a monthly batch: check feed coverage of the batch, train if a trigger fires (matured labels only), then score
  initial              no model in production, retried every MIN_MONTHS_BETWEEN_ATTEMPTS
  feature_outage       champion uses a feed with batch coverage < MIN_GROUP_COVERAGE, train in the same month
  regular_degradation  monitoring RED + enough new cohorts + spacing, live next month
new cohorts and spacing are counted from the last attempt, promoted or not
no model is effective after the last application month (FEATURE_END_DATE)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import config as C
from utils import model_common as M


def in_scope(snapshot_date):
    return C.START_DATE <= snapshot_date <= C.END_DATE


def labelled_cohorts(snapshot_date):
    """number of cohorts with labels on snapshot_date"""
    first, last = C.START_DATE, min(C.month_add(snapshot_date, -C.MOB), C.FEATURE_END_DATE)
    n, d = 0, first
    while d <= last:
        n, d = n + 1, C.month_add(d, 1)
    return n


def _spacing_ok(reg, snapshot_date):
    last = M.last_attempt(reg, snapshot_date, inclusive=False)
    return last is None or M.months_between(last["train_date"], snapshot_date) >= C.MIN_MONTHS_BETWEEN_ATTEMPTS


def pre_inference_trigger(snapshot_date):
    """trigger and reason, decided before scoring"""
    if not in_scope(snapshot_date) or snapshot_date > C.FEATURE_END_DATE:
        return None, "no application batch this month"
    reg = M.read_registry()
    champion = M.production_version(snapshot_date, reg)
    if champion is None:
        if snapshot_date < C.INITIAL_TRAIN_DATE:
            return None, "before the first training date"
        if _spacing_ok(reg, snapshot_date):
            return "initial", "no model in production"
        return None, "initial attempt did not qualify; waiting for the spacing rule (manual review pending)"
    cov = M.batch_coverage(snapshot_date)
    bad = [g for g in M.failing_groups(cov) if g in M.dependent_groups(M.model_entry(champion, reg)["features"])]
    if not bad:
        return None, "champion inputs healthy"
    event = next(e for e in M.outage_events(snapshot_date) if e["group"] == bad[0] and e["end"] is None)
    if event["start"] == snapshot_date:
        return "feature_outage", f"{bad} coverage {cov[bad[0]]:.0%} (first month of the outage event)"
    if _spacing_ok(reg, snapshot_date):
        return "feature_outage", f"{bad} still in outage since {event['start']}; retry after spacing"
    return None, f"{bad} in outage since {event['start']}; within spacing window of the last attempt"


def post_monitoring_trigger(snapshot_date):
    """trigger and reason, decided after monitoring (live next month)"""
    if not in_scope(snapshot_date):
        return None, "out of scope"
    if C.month_add(snapshot_date, 1) > C.FEATURE_END_DATE:
        return None, "a model trained now would be effective after the last application month"
    mon = M.read_gold(C.GOLD_MONITORING, only=snapshot_date)
    if mon.empty:
        return None, "no monitoring result"
    status = mon["overall_status"].iloc[0]
    if status != "RED":
        return None, f"status {status}"
    reg = M.read_registry()
    last = M.last_attempt(reg, snapshot_date, inclusive=True)
    if last is None:
        return None, "no previous attempt"
    new = labelled_cohorts(snapshot_date) - last["n_cohorts"]
    if new < C.MIN_NEW_COHORTS_FOR_RETRAIN:
        return None, f"RED but only {new} new labelled cohorts since the last attempt (< {C.MIN_NEW_COHORTS_FOR_RETRAIN})"
    if M.months_between(last["train_date"], snapshot_date) < C.MIN_MONTHS_BETWEEN_ATTEMPTS:
        return None, "RED but inside the spacing window of the last attempt"
    return "regular_degradation", f"RED with {new} new labelled cohorts since the last attempt"


def should_train_pre_inference(snapshot_date):
    trig, why = pre_inference_trigger(snapshot_date)
    print("pre-inference training decision:", trig, "-", why)
    return trig is not None


def should_train_post_monitoring(snapshot_date):
    trig, why = post_monitoring_trigger(snapshot_date)
    print("post-monitoring training decision:", trig, "-", why)
    return trig is not None

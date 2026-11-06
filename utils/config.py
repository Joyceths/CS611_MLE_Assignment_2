"""settings for the pipeline"""
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

# project root
BASE_DIR = os.environ.get("PROJECT_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# data timeline
START_DATE = "2023-01-01"
FEATURE_END_DATE = "2025-01-01"   # last application month
END_DATE = "2025-07-01"           # cohort 2025-01 gets its label here

# label definition
DPD = 30
MOB = 6

# datamart layout
DATAMART = os.path.join(BASE_DIR, "datamart")
BRONZE_LMS = DATAMART + "/bronze/lms/"
SILVER_LMS = DATAMART + "/silver/loan_daily/"
GOLD_LABEL = DATAMART + "/gold/label_store/"
GOLD_FEATURE = DATAMART + "/gold/feature_store/"
GOLD_PREDICTIONS = DATAMART + "/gold/model_predictions/"
GOLD_MONITORING = DATAMART + "/gold/model_monitoring/"
GOLD_CSI = DATAMART + "/gold/model_monitoring_csi/"
FEATURE_SOURCES = ["attributes", "financials", "clickstream"]
CLICKSTREAM_WINDOW = 6            # months of look-back

# model governance
MODEL_BANK = os.path.join(BASE_DIR, "model_bank") + "/"
REPORT_DIR = os.path.join(BASE_DIR, "monitoring_report") + "/"
INITIAL_TRAIN_DATE = "2024-04-01"  # first month a model may be trained
RANDOM_STATE = 611

# split: train < validation < oot
VALIDATION_COHORTS = 2             # picks algorithm and parameters
OOT_COHORTS = 2                    # newest cohorts: final evaluation and champion comparison

# feature groups that depend on an upstream feed
FEATURE_GROUPS = {"clickstream": lambda c: c.startswith("cs_") or c == "has_clickstream"}
GROUP_COVERAGE_COLUMN = {"clickstream": "has_clickstream"}   # share of applicants with data
MIN_GROUP_COVERAGE = 0.90          # below this is an outage; it ends at >= 0.90

# when to retrain
MIN_NEW_COHORTS_FOR_RETRAIN = 6    # regular retrain: RED + this many new cohorts
MIN_MONTHS_BETWEEN_ATTEMPTS = 2    # spacing between attempts; first month of an outage is exempt

# promotion
MIN_OOT_AUC = 0.75                 # floor for any model going live
MAX_CALIBRATION_GAP = 0.05         # floor: |mean predicted - actual default rate| on OOT
MIN_AUC_GAIN = 0.005               # minimum gain over the champion on the same OOT window

# monitoring thresholds
PSI_AMBER, PSI_RED = 0.10, 0.25    # alerts use PSI vs the OOT reference
CSI_RED_FEATURES = 5                # RED if this many top-20 features have CSI >= PSI_RED
AUC_DROP_AMBER, AUC_DROP_RED = 0.03, 0.06   # drop vs the model's OOT AUC
CALIBRATION_AMBER, CALIBRATION_RED = 0.05, 0.10


def month_add(date_str, months):
    return (datetime.strptime(date_str, "%Y-%m-%d") + relativedelta(months=months)).strftime("%Y-%m-%d")


def tag(date_str):
    return date_str.replace("-", "_")

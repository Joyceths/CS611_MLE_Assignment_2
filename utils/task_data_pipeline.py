"""airflow entry point for the data layers (bronze / silver / gold), one month at a time

usage: python utils/task_data_pipeline.py --layer bronze|silver|gold|validate --snapshotdate 2024-01-01
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config as C

os.chdir(C.BASE_DIR)  # data layer utils use relative paths

import utils.data_processing_bronze_table as bronze
import utils.data_processing_silver_table as silver
import utils.data_processing_gold_table as gold
from utils.spark_utils import get_spark


def main(layer, snapshot_date):
    if not (C.START_DATE <= snapshot_date <= C.END_DATE):
        print(snapshot_date, "is outside the pipeline window", C.START_DATE, "-", C.END_DATE, "- nothing to do")
        return
    has_features = snapshot_date <= C.FEATURE_END_DATE      # no applications after Jan 2025
    spark = get_spark("datamart_" + layer)
    rel = lambda p: os.path.relpath(p, C.BASE_DIR) + "/"

    for d in [C.BRONZE_LMS, C.SILVER_LMS, C.GOLD_LABEL, C.GOLD_FEATURE] + \
             [C.DATAMART + f"/{l}/{s}/" for l in ("bronze", "silver") for s in C.FEATURE_SOURCES]:
        os.makedirs(d, exist_ok=True)

    if layer == "bronze":
        bronze.process_bronze_table(snapshot_date, rel(C.BRONZE_LMS), spark)
        if has_features:
            for s in C.FEATURE_SOURCES:
                bronze.process_bronze_feature_table(s, snapshot_date, rel(C.DATAMART + f"/bronze/{s}"), spark)

    elif layer == "silver":
        silver.process_silver_table(snapshot_date, rel(C.BRONZE_LMS), rel(C.SILVER_LMS), spark)
        if has_features:
            for s in C.FEATURE_SOURCES:
                silver.process_silver_feature_table(s, snapshot_date, rel(C.DATAMART + f"/bronze/{s}"),
                                                    rel(C.DATAMART + f"/silver/{s}"), spark)

    elif layer == "gold":
        # label store
        gold.process_labels_gold_table(snapshot_date, rel(C.SILVER_LMS), rel(C.GOLD_LABEL), spark, C.DPD, C.MOB)
        # feature store
        if has_features:
            gold.process_features_gold_table(
                snapshot_date, rel(C.SILVER_LMS),
                rel(C.DATAMART + "/silver/attributes"), rel(C.DATAMART + "/silver/financials"),
                rel(C.DATAMART + "/silver/clickstream"), rel(C.GOLD_FEATURE), spark,
                cs_long_window=C.CLICKSTREAM_WINDOW)
        else:
            print("no loan applications after", C.FEATURE_END_DATE, "- nothing to build")

    elif layer == "validate":
        gold.check_gold_stores(rel(C.GOLD_FEATURE), rel(C.GOLD_LABEL), spark, C.MOB)
    spark.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", required=True)
    p.add_argument("--snapshotdate", required=True)
    a = p.parse_args()
    main(a.layer, a.snapshotdate)

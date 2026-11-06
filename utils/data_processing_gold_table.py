import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from utils.data_processing_silver_table import check_keys, LOAN_TYPES

def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "loan_start_date", "label", "label_def", "snapshot_date")

    # data quality check
    check_keys(df, ["loan_id"], "gold_label_store")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


# feature store: one row per loan application, only data up to the application date
OCCUPATIONS = ["Accountant", "Architect", "Developer", "Doctor", "Engineer", "Entrepreneur", "Journalist",
               "Lawyer", "Manager", "Mechanic", "Media_Manager", "Musician", "Scientist", "Teacher", "Writer"]
# start at 1, so 0 = unknown
CREDIT_MIX_ORDINAL = {"Bad": 1, "Standard": 2, "Good": 3}
PAYMENT_VALUE_ORDINAL = {"Small": 1, "Medium": 2, "Large": 3}
NUMERIC_FEATURES = {
    "Annual_Income": "annual_income", "Monthly_Inhand_Salary": "monthly_inhand_salary",
    "Num_Bank_Accounts": "num_bank_accounts", "Num_Credit_Card": "num_credit_card",
    "Interest_Rate": "interest_rate", "Num_of_Loan": "num_of_loan",
    "Delay_from_due_date": "delay_from_due_date", "Num_of_Delayed_Payment": "num_of_delayed_payment",
    "Changed_Credit_Limit": "changed_credit_limit", "Num_Credit_Inquiries": "num_credit_inquiries",
    "Outstanding_Debt": "outstanding_debt", "Credit_Utilization_Ratio": "credit_utilization_ratio",
    "Credit_History_Age_Months": "credit_history_age_months", "Total_EMI_per_month": "total_emi_per_month",
    "Amount_invested_monthly": "amount_invested_monthly", "Monthly_Balance": "monthly_balance",
}
# columns that can be null after silver cleaning -> add _missing flag
NULLABLE_FEATURES = ["age", "annual_income", "num_bank_accounts", "num_credit_card", "interest_rate",
                     "num_of_delayed_payment", "changed_credit_limit", "num_credit_inquiries",
                     "total_emi_per_month", "amount_invested_monthly", "monthly_balance"]
CLICKSTREAM_COLS = ["fe_" + str(i) for i in range(1, 21)]


def snake(name):
    return name.lower().replace("-", "_").replace(" ", "_")


def to_ordinal(column_name, mapping):
    expr = F.lit(None)
    for k, v in mapping.items():
        expr = F.when(col(column_name) == k, F.lit(v)).otherwise(expr)
    return expr.cast(IntegerType())


def ratio(num, den):
    return F.when(den > 0, num / den).cast(FloatType())


def process_features_gold_table(snapshot_date_str, silver_loan_daily_directory, silver_attributes_directory, silver_financials_directory, silver_clickstream_directory, gold_feature_store_directory, spark, cs_long_window=12):
    date_str = snapshot_date_str.replace('-','_')

    # population: loans at installment 0 (= application)
    lms = spark.read.parquet(silver_loan_daily_directory + "silver_loan_daily_" + date_str + ".parquet")
    apps = lms.filter(col("installment_num") == 0) \
              .select("loan_id", "Customer_ID", col("loan_start_date").alias("snapshot_date"))
    n_apps = apps.count()

    # attributes and financials at application date
    attr = spark.read.parquet(silver_attributes_directory + "silver_attributes_" + date_str + ".parquet")
    fin = spark.read.parquet(silver_financials_directory + "silver_financials_" + date_str + ".parquet")
    n_no_attr = apps.join(attr, ["Customer_ID", "snapshot_date"], "left_anti").count()
    n_no_fin = apps.join(fin, ["Customer_ID", "snapshot_date"], "left_anti").count()
    if n_no_attr > 0 or n_no_fin > 0:
        raise ValueError(f"gold_feature_store {snapshot_date_str}: {n_no_attr} applications without attributes, {n_no_fin} without financials")
    df = apps.join(attr, ["Customer_ID", "snapshot_date"], "left").join(fin, ["Customer_ID", "snapshot_date"], "left")

    # feature: attributes
    df = df.withColumn("age", col("Age").cast(FloatType()))
    for occ in OCCUPATIONS:
        df = df.withColumn("occupation_" + snake(occ), (col("Occupation") == occ).cast(IntegerType()))
    df = df.withColumn("occupation_unknown", col("Occupation").isNull().cast(IntegerType()))

    # feature: financials numeric
    for src, dst in NUMERIC_FEATURES.items():
        df = df.withColumn(dst, col(src).cast(FloatType()))

    # feature: encode categories as numbers
    df = df.withColumn("credit_mix_ordinal", to_ordinal("Credit_Mix", CREDIT_MIX_ORDINAL))
    df = df.withColumn("credit_mix_unknown", col("Credit_Mix").isNull().cast(IntegerType()))
    df = df.withColumn("pay_min_amount_yes", (col("Payment_of_Min_Amount") == "Yes").cast(IntegerType()))
    df = df.withColumn("pay_min_amount_unknown", col("Payment_of_Min_Amount").isNull().cast(IntegerType()))
    df = df.withColumn("spent_level_high", (col("Spent_Level") == "High").cast(IntegerType()))
    df = df.withColumn("payment_value_ordinal", to_ordinal("Payment_Value_Level", PAYMENT_VALUE_ORDINAL))
    df = df.withColumn("payment_behaviour_unknown", col("Payment_Behaviour").isNull().cast(IntegerType()))
    for lt in LOAN_TYPES:
        df = df.withColumn("loan_type_" + snake(lt), F.array_contains(col("Type_of_Loan_list"), lt).cast(IntegerType()))

    # feature: ratios
    df = df.withColumn("debt_to_annual_income", ratio(col("outstanding_debt"), col("annual_income")))
    df = df.withColumn("emi_to_monthly_salary", ratio(col("total_emi_per_month"), col("monthly_inhand_salary")))
    df = df.withColumn("invested_to_monthly_salary", ratio(col("amount_invested_monthly"), col("monthly_inhand_salary")))
    df = df.withColumn("delayed_payments_per_loan", ratio(col("num_of_delayed_payment"), col("num_of_loan")))
    # ratio undefined when num_of_loan = 0 -> add flag
    df = df.withColumn("delayed_payments_per_loan_missing", col("delayed_payments_per_loan").isNull().cast(IntegerType()))

    # feature: _missing flags + count of invalid fields
    for c in NULLABLE_FEATURES:
        df = df.withColumn(c + "_missing", col(c).isNull().cast(IntegerType()))
    df = df.withColumn("dq_num_invalid_fields", sum(col(c + "_missing") for c in NULLABLE_FEATURES))

    # feature: clickstream up to the application month (no future data)
    # value at month T, 3 month avg and long window avg (only months that exist)
    # no clickstream (Jul 2024 onwards) -> 0 and has_clickstream = 0
    long_w = cs_long_window
    lookback_dates = [(datetime.strptime(snapshot_date_str, "%Y-%m-%d") - relativedelta(months=i)).strftime("%Y_%m_%d") for i in range(max(3, long_w))]
    cs_files = [silver_clickstream_directory + "silver_clickstream_" + d + ".parquet" for d in lookback_dates]
    cs_files = [f for f in cs_files if os.path.exists(f)]
    cs_cols = (["cs_" + c + "_latest" for c in CLICKSTREAM_COLS] + ["cs_" + c + "_avg_3m" for c in CLICKSTREAM_COLS]
               + ["cs_" + c + "_avg_" + str(long_w) + "m" for c in CLICKSTREAM_COLS])
    months_cols = ["cs_months_observed_3m", "cs_months_observed_" + str(long_w) + "m"]
    if cs_files:
        t = F.lit(snapshot_date_str).cast(DateType())
        cs = spark.read.parquet(*cs_files).filter(col("snapshot_date") <= t)
        cs = cs.withColumn("in_3m", col("snapshot_date") > F.add_months(t, -3))
        cs_latest = cs.filter(col("snapshot_date") == t) \
                      .select("Customer_ID", *[col(c).alias("cs_" + c + "_latest") for c in CLICKSTREAM_COLS])
        cs_avg = cs.groupBy("Customer_ID").agg(
            *[F.avg(F.when(col("in_3m"), col(c))).cast(FloatType()).alias("cs_" + c + "_avg_3m") for c in CLICKSTREAM_COLS],
            *[F.avg(c).cast(FloatType()).alias("cs_" + c + "_avg_" + str(long_w) + "m") for c in CLICKSTREAM_COLS],
            F.countDistinct(F.when(col("in_3m"), col("snapshot_date"))).cast(IntegerType()).alias(months_cols[0]),
            F.countDistinct("snapshot_date").cast(IntegerType()).alias(months_cols[1]))
        df = df.join(cs_avg, "Customer_ID", "left").join(cs_latest, "Customer_ID", "left")
    else:
        for c in cs_cols:
            df = df.withColumn(c, F.lit(None).cast(FloatType()))
        for c in months_cols:
            df = df.withColumn(c, F.lit(None).cast(IntegerType()))
    for c in months_cols:
        df = df.withColumn(c, F.coalesce(col(c), F.lit(0)).cast(IntegerType()))
    df = df.withColumn("has_clickstream", (col(months_cols[1]) > 0).cast(IntegerType()))

    # fill missing with 0 (flags added above)
    feature_cols = (
        ["age"] + ["occupation_" + snake(o) for o in OCCUPATIONS] + ["occupation_unknown"]
        + list(NUMERIC_FEATURES.values())
        + ["credit_mix_ordinal", "credit_mix_unknown", "pay_min_amount_yes", "pay_min_amount_unknown",
           "spent_level_high", "payment_value_ordinal", "payment_behaviour_unknown"]
        + ["loan_type_" + snake(lt) for lt in LOAN_TYPES]
        + ["debt_to_annual_income", "emi_to_monthly_salary", "invested_to_monthly_salary", "delayed_payments_per_loan",
           "delayed_payments_per_loan_missing"]
        + [c + "_missing" for c in NULLABLE_FEATURES] + ["dq_num_invalid_fields"]
        + ["has_clickstream"] + months_cols + cs_cols
    )
    df = df.select("loan_id", "Customer_ID", "snapshot_date", *feature_cols).fillna(0, subset=feature_cols)

    # data quality check
    if df.count() != n_apps:
        raise ValueError(f"gold_feature_store {snapshot_date_str}: expected {n_apps} rows, got {df.count()}")
    check_keys(df, ["loan_id"], "gold_feature_store")
    n_bad = df.filter(F.greatest(*[(col(c).isNull() | F.isnan(col(c)) | (F.abs(col(c)) == float("inf"))).cast(IntegerType())
                                   for c in feature_cols]) == 1).count()
    if n_bad > 0:
        raise ValueError(f"gold_feature_store {snapshot_date_str}: {n_bad} rows with NULL / NaN / infinite features")

    # save gold table - IRL connect to database to write
    partition_name = "gold_feature_store_" + date_str + ".parquet"
    filepath = gold_feature_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath, 'rows:', df.count(), 'features:', len(feature_cols))

    return df

# check feature store and label store match
def check_gold_stores(gold_feature_store_directory, gold_label_store_directory, spark, mob):
    features = spark.read.parquet(*glob.glob(gold_feature_store_directory + "*"))
    labels = spark.read.parquet(*glob.glob(gold_label_store_directory + "*"))
    n_loans = spark.read.csv("data/lms_loan_daily.csv", header=True).select("loan_id").distinct().count()

    # every loan has one feature row and one label
    check_keys(features, ["loan_id"], "gold_feature_store (all months)")
    check_keys(labels, ["loan_id"], "gold_label_store (all months)")
    joined = labels.join(features.select("loan_id", col("snapshot_date").alias("feature_date")), "loan_id", "inner")
    if not (features.count() == labels.count() == joined.count() == n_loans):
        raise ValueError("feature store and label store do not cover every loan exactly once")

    # label date = application date + 6 months
    if joined.filter((col("feature_date") != col("loan_start_date")) |
                     (col("snapshot_date") != F.add_months(col("feature_date"), mob))).count() > 0:
        raise ValueError("label is not observed exactly MOB months after the feature snapshot")

    print("feature store row_count:", features.count(), "columns:", len(features.columns))
    print("label store row_count:", labels.count())
    print("joined row_count:", joined.count())

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

# check primary key, stop if key is missing or duplicated
def check_keys(df, keys, table):
    null_filter = F.lit(False)
    for k in keys:
        null_filter = null_filter | col(k).isNull() | (F.trim(col(k).cast("string")) == "")
    n_null = df.filter(null_filter).count()
    if n_null > 0:
        raise ValueError(f"{table}: {n_null} rows with missing key {keys}")
    n_duplicates = df.count() - df.select(*keys).distinct().count()
    if n_duplicates > 0:
        raise ValueError(f"{table}: {n_duplicates} duplicate rows on key {keys}")


def process_silver_table(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0, subset=["installments_missed"])
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # data quality check
    check_keys(df, ["loan_id", "snapshot_date"], "silver_loan_daily")

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


# feature tables, fixed cleaning rules only, no mean / median
LOAN_TYPES = ["Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan", "Home Equity Loan", "Mortgage Loan", "Payday Loan", "Personal Loan", "Student Loan", "Not Specified"]


def clean_number(column_name):
    return F.regexp_replace(F.trim(col(column_name)), "_", "")


def valid_range(expr, low, high):
    return F.when((expr >= low) & (expr <= high), expr)


def process_silver_feature_table(source, snapshot_date_str, bronze_directory, silver_directory, spark):
    # connect to bronze table
    partition_name = "bronze_" + source + "_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=False)
    print('loaded from:', filepath, 'row count:', df.count())

    df = df.withColumn("Customer_ID", F.trim(col("Customer_ID"))).withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    if source == "clickstream":
        fe_cols = ["fe_" + str(i) for i in range(1, 21)]
        for c in fe_cols:
            df = df.withColumn(c, clean_number(c).cast(IntegerType()))
        df = df.select("Customer_ID", "snapshot_date", *fe_cols)

    elif source == "attributes":
        # clean data: "28_" -> 28, keep age 14-100 (normal 14-56, bad values -500 or 169+)
        df = df.withColumn("Age", valid_range(clean_number("Age").cast(IntegerType()), 14, 100))
        df = df.withColumn("Occupation", F.when(col("Occupation").rlike("^_+$") | (F.trim(col("Occupation")) == ""), None)
                                          .otherwise(F.trim(col("Occupation"))))
        # drop PII (Name, SSN)
        df = df.select("Customer_ID", "snapshot_date", "Age", "Occupation")
    
    elif source == "financials":
        # clean data: remove "_" and set data types
        float_cols = ["Annual_Income", "Monthly_Inhand_Salary", "Changed_Credit_Limit", "Outstanding_Debt", "Credit_Utilization_Ratio", "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance"]
        int_cols = ["Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment", "Num_Credit_Inquiries"]
        for c in float_cols:
            df = df.withColumn(c, clean_number(c).cast(FloatType()))
        for c in int_cols:
            df = df.withColumn(c, clean_number(c).cast(FloatType()).cast(IntegerType()))

        # clean data: placeholder values -> null
        df = df.withColumn("Amount_invested_monthly", F.when(col("Amount_invested_monthly") < 10000, col("Amount_invested_monthly")))
        df = df.withColumn("Monthly_Balance", F.when(F.abs(col("Monthly_Balance")) <= 1e9, col("Monthly_Balance")))

        # clean data: impossible values -> null, each limit sits in the gap between normal and bad values
        # e.g inquiries normal 0-17, bad from 26; delayed payments normal 0-28, bad from 47
        df = df.withColumn("Num_Bank_Accounts", valid_range(col("Num_Bank_Accounts"), 0, 20))
        df = df.withColumn("Num_Credit_Card", valid_range(col("Num_Credit_Card"), 0, 20))
        df = df.withColumn("Interest_Rate", valid_range(col("Interest_Rate"), 0, 50))
        df = df.withColumn("Num_of_Delayed_Payment", valid_range(col("Num_of_Delayed_Payment"), 0, 30))
        df = df.withColumn("Num_Credit_Inquiries", valid_range(col("Num_Credit_Inquiries"), 0, 20))
        # clean data: check income and EMI against monthly salary where normal income <= 1.9x of 12 salaries, bad from 4.9x; EMI above salary -> null
        df = df.withColumn("Annual_Income", F.when(col("Annual_Income") <= 2 * 12 * col("Monthly_Inhand_Salary"), col("Annual_Income")))
        df = df.withColumn("Total_EMI_per_month", F.when(col("Total_EMI_per_month") <= col("Monthly_Inhand_Salary"), col("Total_EMI_per_month")))

        # Type_of_Loan: "Auto Loan, and Payday Loan" -> ["Auto Loan", "Payday Loan"]
        loan_list = F.when(col("Type_of_Loan").isNull(), F.array().cast("array<string>")) \
                     .otherwise(F.split(F.regexp_replace(col("Type_of_Loan"), r",\s*and\s+|,\s*", ","), ","))
        df = df.withColumn("Type_of_Loan_list", F.filter(F.transform(loan_list, lambda x: F.trim(x)), lambda x: x != ""))
        # Num_of_Loan outside 0-15 -> number of loan types (they match in all valid rows)
        df = df.withColumn("Num_of_Loan", F.when(col("Num_of_Loan").between(0, 15), col("Num_of_Loan"))
                                           .otherwise(F.size(col("Type_of_Loan_list"))).cast(IntegerType()))

        # clean data: garbage categories "_", "NM", "!@9#%8" -> null
        df = df.withColumn("Credit_Mix", F.when(col("Credit_Mix").isin("Good", "Standard", "Bad"), col("Credit_Mix")))
        df = df.withColumn("Payment_of_Min_Amount", F.when(col("Payment_of_Min_Amount").isin("Yes", "No"), col("Payment_of_Min_Amount")))
        df = df.withColumn("Payment_Behaviour", F.when(col("Payment_Behaviour").rlike("^(Low|High)_spent_(Small|Medium|Large)_value_payments$"), col("Payment_Behaviour")))

        # augment data: "10 Years and 9 Months" -> 129 months
        years = F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s*Years", 1).cast(IntegerType())
        months = F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s*Months", 1).cast(IntegerType())
        df = df.withColumn("Credit_History_Age_Months", (years * 12 + months).cast(IntegerType()))
        # augment data: split Payment_Behaviour into spent level and payment size
        df = df.withColumn("Spent_Level", F.regexp_extract(col("Payment_Behaviour"), r"^(Low|High)_spent", 1))
        df = df.withColumn("Payment_Value_Level", F.regexp_extract(col("Payment_Behaviour"), r"_(Small|Medium|Large)_value", 1))
        for c in ["Spent_Level", "Payment_Value_Level"]:
            df = df.withColumn(c, F.when(col(c) != "", col(c)))

        df = df.select(
            "Customer_ID", "snapshot_date",
            "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
            "Num_of_Loan", "Type_of_Loan_list", "Delay_from_due_date", "Num_of_Delayed_Payment",
            "Changed_Credit_Limit", "Num_Credit_Inquiries", "Credit_Mix", "Outstanding_Debt", "Credit_Utilization_Ratio",
            "Credit_History_Age_Months", "Payment_of_Min_Amount", "Total_EMI_per_month", "Amount_invested_monthly",
            "Payment_Behaviour", "Spent_Level", "Payment_Value_Level", "Monthly_Balance",
        )

    # data quality check
    check_keys(df, ["Customer_ID", "snapshot_date"], "silver_" + source)

    # save silver table - IRL connect to database to write
    partition_name = "silver_" + source + "_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df

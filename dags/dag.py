"""loan default ml pipeline: data -> coverage check / train -> inference -> monitoring
one dag run = one monthly snapshot, backfill 2023-01 to 2025-07
"""
import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, ShortCircuitOperator

PROJECT_DIR = os.environ.get("PROJECT_DIR", "/opt/airflow/project")
sys.path.insert(0, PROJECT_DIR)
from utils import config as C

PY = os.environ.get("PROJECT_PYTHON", "python")
RUN = f"cd {PROJECT_DIR} && {PY} utils/"
DS = '--snapshotdate "{{ ds }}"'


def _pre_inference_branch(ds, **_):
    """initial or feature outage training, before scoring"""
    from utils import model_governance as G
    return "train_before_scoring" if G.should_train_pre_inference(ds) else "skip_train_before_scoring"


def _post_monitoring_branch(ds, **_):
    """regular retraining after monitoring"""
    from utils import model_governance as G
    return "train_after_monitoring" if G.should_train_post_monitoring(ds) else "no_retrain"


def _is_last_month(ds, **_):
    return ds == C.END_DATE


default_args = {"owner": "joyce", "depends_on_past": True, "retries": 1, "retry_delay": timedelta(minutes=1)}

with DAG(
    dag_id="loan_default_ml_pipeline",
    description="Monthly datamart build, model training / selection, inference and monitoring",
    default_args=default_args,
    schedule="0 0 1 * *",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2025, 7, 1),
    catchup=True,
    max_active_runs=1,
    tags=["cs611", "mle", "loan-default"],
) as dag:

    start = EmptyOperator(task_id="start")

    # data pipeline
    bronze = BashOperator(task_id="bronze_tables", bash_command=f"{RUN}task_data_pipeline.py --layer bronze {DS}")
    silver = BashOperator(task_id="silver_tables", bash_command=f"{RUN}task_data_pipeline.py --layer silver {DS}")
    gold = BashOperator(task_id="gold_label_and_feature_store", bash_command=f"{RUN}task_data_pipeline.py --layer gold {DS}")
    data_ready = EmptyOperator(task_id="data_ready")

    last_month_gate = ShortCircuitOperator(task_id="only_on_last_month_for_validation", python_callable=_is_last_month)
    validate = BashOperator(task_id="validate_gold_stores", bash_command=f"{RUN}task_data_pipeline.py --layer validate {DS}")

    # check coverage, train if needed, before scoring
    pre_check = BranchPythonOperator(task_id="training_check_before_scoring", python_callable=_pre_inference_branch)
    train_pre = BashOperator(task_id="train_before_scoring",
                             bash_command=f"{RUN}model_train.py {DS} --mode pre_inference")
    skip_pre = EmptyOperator(task_id="skip_train_before_scoring")
    model_ready = EmptyOperator(task_id="model_ready", trigger_rule="none_failed_min_one_success")

    # inference and monitoring
    inference = BashOperator(task_id="model_inference", bash_command=f"{RUN}model_inference.py {DS}")
    monitoring = BashOperator(task_id="model_monitoring", bash_command=f"{RUN}model_monitor.py {DS}")
    last_month_gate_plots = ShortCircuitOperator(task_id="only_on_last_month_for_plots", python_callable=_is_last_month)
    plots = BashOperator(task_id="monitoring_visualisation", bash_command=f"{RUN}plot_monitoring.py {DS}")

    # retrain check after monitoring
    retrain_check = BranchPythonOperator(task_id="retrain_check", python_callable=_post_monitoring_branch)
    train_post = BashOperator(task_id="train_after_monitoring",
                              bash_command=f"{RUN}model_train.py {DS} --mode post_monitoring")
    no_retrain = EmptyOperator(task_id="no_retrain")
    end = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    start >> bronze >> silver >> gold >> data_ready
    gold >> last_month_gate >> validate
    data_ready >> pre_check >> [train_pre, skip_pre] >> model_ready
    model_ready >> inference >> monitoring >> retrain_check >> [train_post, no_retrain] >> end
    monitoring >> last_month_gate_plots >> plots

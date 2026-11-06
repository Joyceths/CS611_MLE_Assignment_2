# Airflow + Java (for PySpark) + the project's python dependencies
FROM apache/airflow:2.10.5-python3.11

USER root
ENV DEBIAN_FRONTEND=noninteractive

# Java 17 for Spark, libgomp for xgboost, procps for Spark's scripts
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jre-headless libgomp1 procps bash && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /bin/bash /bin/sh && \
    mkdir -p /usr/lib/jvm/java-17-openjdk-amd64/bin && \
    ln -sf "$(readlink -f "$(which java)")" /usr/lib/jvm/java-17-openjdk-amd64/bin/java

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin
ENV PROJECT_DIR=/opt/airflow/project

COPY requirements.txt /tmp/requirements.txt
USER airflow
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# project code (also bind-mounted by docker-compose so outputs appear in the local folders)
WORKDIR /opt/airflow/project
COPY --chown=airflow:root dags/ /opt/airflow/dags/
COPY --chown=airflow:root utils/ /opt/airflow/project/utils/

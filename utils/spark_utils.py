import pyspark


def get_spark(app="ml_pipeline"):
    spark = (pyspark.sql.SparkSession.builder.appName(app).master("local[2]")
             .config("spark.ui.enabled", "false")
             .config("spark.sql.shuffle.partitions", "4")
             .config("spark.driver.memory", "2g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark

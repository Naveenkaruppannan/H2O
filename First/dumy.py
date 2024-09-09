from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder \
    .appName('Basic Test') \
    .getOrCreate()

# Create a simple DataFrame
df = spark.createDataFrame([(1, "test"), (2, "example")], ["id", "value"])
df.show()

# Stop SparkSession
spark.stop()

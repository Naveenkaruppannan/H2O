from pyspark.sql import SparkSession
import h2o
from pysparkling import H2OContext

try:
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("Sparkling Water Test") \
        .getOrCreate()

    print("SparkSession created successfully")

    # Initialize H2OContext
    h2o_context = H2OContext.getOrCreate(spark)
    print("H2OContext created successfully")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Stop SparkSession
    spark.stop()

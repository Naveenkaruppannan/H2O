import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import col, year, month, dayofmonth
from pyspark.sql.types import DateType

# Set environment variables
os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk1.8.0_202'
os.environ['SPARK_HOME'] = r'C:\spark\spark-3.5.2-bin-hadoop3\spark-3.5.2-bin-hadoop3'

# Create a SparkSession
spark = SparkSession.builder \
    .appName('Data Preprocessing') \
    .config('spark.pyspark.python', r'C:/Users/vaf/AppData/Local/Programs/Python/Python312/python.exe') \
    .config('spark.pyspark.driver.python', r'C:/Users/vaf/AppData/Local/Programs/Python/Python312/python.exe') \
    .getOrCreate()

# Read the CSV file into a DataFrame
df = spark.read.csv("financial_portfolio_data.csv", header=True, inferSchema=True)

# Rename columns to match expected schema
df = df.withColumnRenamed('Date', 'Date') \
       .withColumnRenamed('Asset', 'Asset') \
       .withColumnRenamed('Price', 'Price')

# Convert 'Date' column to DateType
df = df.withColumn('Date', col('Date').cast(DateType()))

# Extract Year, Month, and Day from 'Date'
df = df.withColumn('Year', year(col('Date')))
df = df.withColumn('Month', month(col('Date')))
df = df.withColumn('Day', dayofmonth(col('Date')))

# Drop the original 'Date' column
df = df.drop('Date')

# String Indexing
indexer = StringIndexer(inputCol='Asset', outputCol='AssetIndex')
df_indexed = indexer.fit(df).transform(df)

# One-Hot Encoding
encoder = OneHotEncoder(inputCols=['AssetIndex'], outputCols=['AssetVec'])
df_encoded = encoder.fit(df_indexed).transform(df_indexed)

# Vector Assembling
assembler = VectorAssembler(
    inputCols=['AssetVec', 'Price', 'Year', 'Month', 'Day'],
    outputCol='features'
)
df_final = assembler.transform(df_encoded)

# Show the final DataFrame
df_final.show(truncate=False)

# Stop the SparkSession
spark.stop()

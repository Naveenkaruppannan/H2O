from pyspark.sql import SparkSession
from pyspark.sql.functions import col,lit

# Create a SparkSession
spark = SparkSession.builder \
    .appName('Spark data preprocessing') \
    .getOrCreate()  # Correct method name

# Read a CSV file into a DataFrame
df = spark.read.csv("financial_portfolio_data.csv", header=True, inferSchema=True)

# Show the first few rows of the DataFrame
df.show() 

# Filter the Price Greather the 100

filter_Price=df.filter(df['Price']>100)
filter_Price.show()

# # Filter the Bond A

filter_Asset=df.filter(df['Asset']== 'Bond A')
filter_Asset.show()

df_filtered_date_range = df.filter((col('Date') >= lit('2023-01-01')) & (col('Date') <= lit('2023-12-31')))
df_filtered_date_range.show()

# df.show()

Check_null=df.filter(df['Price'].isNull())
Check_null.show()

# Adding a new column with a constant value

df_with_literal = df.withColumn('NewColumn', lit(100))
df_with_literal.show()


df.printSchema()

# Stop the SparkSession when done
spark.stop()

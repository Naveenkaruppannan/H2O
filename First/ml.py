from pyspark.sql import SparkSession
from pysparkling import H2OContext
from h2o.estimators import H2OGradientBoostingEstimator


# Initialize SparkSession
spark = SparkSession.builder.appName("CategoricalDataExample").getOrCreate()

# Initialize H2OContext
hc = H2OContext.getOrCreate(spark)
print("H2O Context initialized.")

# Load data into Spark DataFrame
df = spark.read.csv("E:\\Python\\H2O\\First\\ai_job_market_insights.csv", header=True, inferSchema=True)

# Convert Spark DataFrame to H2OFrame
hf = hc.as_h2o_frame(df)

# Define and train the model
model = H2OGradientBoostingEstimator()
model.train(x=list(df.columns), y="Job_Growth_Projection", training_frame=hf)

# Make predictions
predictions = model.predict(hf)

# Convert predictions back to Spark DataFrame
df_predictions = hc.as_spark_frame(predictions)

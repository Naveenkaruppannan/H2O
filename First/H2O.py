import h2o
from h2o.estimators import H2OGradientBoostingEstimator

# Initialize H2O cluster
h2o.init()

# Import dataset
data_path = "ai_job_market_insights.csv"
data = h2o.import_file(data_path)

# Print the column names to find the correct response column
print("Column names in the dataset:", data.columns)

# Print the first few rows of the dataset
print(data.head())

# Split the dataset into training and validation sets
train, valid = data.split_frame(ratios=[0.8], seed=1234)

# Define the response and predictor variables
response = "Job_Growth_Projection"  
predictors = ["Job_Title", "Industry", "Company_Size", "Location", "AI_Adoption_Level", 
              "Automation_Risk", "Required_Skills", "Salary_USD", "Remote_Friendly"]

# Initialize and train the model
gbm = H2OGradientBoostingEstimator()
gbm.train(x=predictors, y=response, training_frame=train, validation_frame=valid)

# Print model performance
performance = gbm.model_performance(valid)
print(performance)

# Make predictions on a sample
sample = valid[0, predictors]
prediction = gbm.predict(sample)
print(prediction)

# Shutdown the H2O cluster
h2o.shutdown(prompt=False)

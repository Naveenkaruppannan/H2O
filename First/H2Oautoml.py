import h2o
from h2o.automl import H2OAutoML
import pandas as pd

# Initialize H2O cluster
h2o.init()

# Load your dataset (replace 'your_data.csv' with your file path)
file_path = 'ai_job_market_insights.csv'
data = pd.read_csv(file_path)

# Convert to H2O Frame
h2o_data = h2o.H2OFrame(data)

# View the data to see the types of columns
print(h2o_data.head())
print(h2o_data.types)

# Specify the target and feature columns
# Replace 'target_column' with the name of your target variable
target = 'Job_Growth_Projection'
features = [col for col in h2o_data.columns if col != target]

# Split the data into training and validation sets
train, valid = h2o_data.split_frame(ratios=[0.8], seed=1234)

# Initialize and train H2O AutoML
aml = H2OAutoML(max_runtime_secs=600, seed=1234)  # Adjust max_runtime_secs as needed
aml.train(x=features, y=target, training_frame=train, validation_frame=valid)

# View the AutoML Leaderboard
lb = aml.leaderboard
print(lb)

# Get the best model
best_model = aml.leader

# Make predictions
preds = best_model.predict(valid)

# View predictions
print(preds.head())

# Show categorical encoding information
print("Column types after AutoML processing:")
print(h2o_data.types)

# Shutdown the H2O cluster
h2o.shutdown(prompt=False)

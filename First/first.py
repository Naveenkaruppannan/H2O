import h2o
from h2o.automl import H2OAutoML

# Initialize the H2O cluster
h2o.init()

# Import the CSV file into H2OFrame
file_path = 'ai_job_market_insights.csv'
h2o_frame = h2o.import_file(file_path)

# Print the types of columns to understand their nature
print("Column Types:")
print(h2o_frame.types)

# Check and print categorical columns
print("\nCategorical Columns:")
for col in h2o_frame.columns:
    if h2o_frame[col].type == 'enum':
        print(f'Column {col} is categorical (enum).')

# Define the target variable and feature columns
target = 'Job_Growth_Projection'  
features = [col for col in h2o_frame.columns if col != target]

# Split the data into training and test sets
train, test = h2o_frame.split_frame(ratios=[0.8], seed=1234)

# Initialize and train H2O AutoML
aml = H2OAutoML(max_runtime_secs=3600, seed=1)  # Set max_runtime_secs as needed
aml.train(x=features, y=target, training_frame=train)

# Get the leaderboard of models
leaderboard = aml.leaderboard
print("\nLeaderboard:")
print(leaderboard)

# Evaluate the model performance on the test set
performance = aml.leader.model_performance(test_data=test)
print("\nModel Performance:")
print(performance)

# Shutdown the H2O cluster using the recommended method
h2o.cluster().shutdown()

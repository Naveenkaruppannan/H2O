import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import mlflow
import mlflow.h2o

# Initialize H2O cluster
h2o.init()

# Load your dataset
file_path = 'ai_job_market_insights.csv'
data = pd.read_csv(file_path)

# Convert to H2O Frame
h2o_data = h2o.H2OFrame(data)

# Specify the target and feature columns
target = 'Job_Growth_Projection'
features = [col for col in h2o_data.columns if col != target]

# Split the data into training and validation sets
train, valid = h2o_data.split_frame(ratios=[0.8], seed=1234)

# Initialize MLflow experiment
mlflow.set_experiment("H2O_AutoML_Job_Growth_Projection")

# Start MLflow run
with mlflow.start_run():
    
    # Initialize and train H2O AutoML
    aml = H2OAutoML(max_runtime_secs=600, seed=1234)
    aml.train(x=features, y=target, training_frame=train, validation_frame=valid)
    
    # View the AutoML Leaderboard
    lb = aml.leaderboard
    print(lb)
    
    # Get the best model
    best_model = aml.leader
    
    # Log the model with MLflow
    mlflow.h2o.log_model(best_model, "model")
    
    # Make predictions on validation data
    preds = best_model.predict(valid)
    print(preds.head())
    
    # Log predictions and metrics with MLflow
    mlflow.log_param("max_runtime_secs", 600)
    mlflow.log_param("seed", 1234)
    
    # Show categorical encoding information
    print("Column types after AutoML processing:")
    print(h2o_data.types)

# Function to handle user input and make predictions
def predict_user_input(user_input):
    # Convert user input to DataFrame
    user_input_df = pd.DataFrame([user_input])
    
    # Convert to H2O Frame
    user_input_h2o = h2o.H2OFrame(user_input_df)
    
    # Make predictions
    predictions = best_model.predict(user_input_h2o)
    
    # Display predictions
    print("Predictions for user input:")
    print(predictions)

# Function to get user input from terminal
def get_user_input():
    print("Please enter the following details:")
    user_input = {
        'Job_Title': input("Job Title: "),
        'Industry': input("Industry: "),
        'Company_Size': input("Company Size: "),
        'Location': input("Location: "),
        'AI_Adoption_Level': input("AI Adoption Level: "),
        'Automation_Risk': input("Automation Risk: "),
        'Required_Skills': input("Required Skills: "),
        'Salary_USD': float(input("Salary (USD): ")),
        'Remote_Friendly': input("Remote Friendly (Yes/No): ")
    }
    return user_input

# Collect user input
user_input = get_user_input()

# Predict based on user input
predict_user_input(user_input)

# Shutdown the H2O cluster
h2o.shutdown(prompt=False)

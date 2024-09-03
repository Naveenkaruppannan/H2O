import h2o
from h2o.frame import H2OFrame

# Initialize the H2O cluster
h2o.init()

# Import a CSV file into H2OFrame (replace 'path/to/your/data.csv' with your actual file path)
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

# Shutdown the H2O cluster using the recommended method
h2o.cluster().shutdown()

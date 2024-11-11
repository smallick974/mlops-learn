from azureml.core import Workspace, Dataset, Experiment

# Load the workspace from the saved config file
ws = Workspace.from_config()

# Create or get an existing experiment 
experiment = Experiment(workspace=ws, name='data-preprocessing')

# Start a new run 
run = experiment.start_logging()

# Access the registered dataset
processed_data = Dataset.get_by_name(workspace=ws, name='iris_data_set')

# Load the dataset into a pandas DataFrame
df = processed_data.to_pandas_dataframe()

run.log('dataset_shape', df.shape)
run.log('dataset_species_count', df['Species'].value_counts())
run.log('dataset_null_sum', df.isnull().sum())

# Step 4: Complete the run 
run.complete()

print(df.head(2))
print(df.info())
print(df.shape)
print(df['Species'].value_counts())
print(df.isnull().sum())

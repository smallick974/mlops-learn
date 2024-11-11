from azureml.core import Workspace, Datastore
from azureml.core.dataset import Dataset

import sys 
import os 

# Get the root directory (one level up from the current file) 
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 

# Add the root directory to sys.path
sys.path.append(root_dir)

from config import account_key, account_name, container_name, datastore_name

datastore_name = datastore_name
container_name = container_name
account_name = account_name
account_key = account_key

# Connect to your Azure ML workspace
ws = Workspace.from_config()

# Register the blob storage account as a datastore
datastore = Datastore.register_azure_blob_container(
    workspace = ws,
    datastore_name = datastore_name,
    container_name = container_name,
    account_name = account_name,
    account_key = account_key,
    create_if_not_exists = True
)

# Define the path to the data in the datastore
datastore_path = [(datastore, 'iris.csv')]

# Create a TabularDataset from the data in the datastore
dataset = Dataset.Tabular.from_delimited_files(path=datastore_path)

# Register the dataset in the workspace (optional)
dataset = dataset.register(
    workspace=ws,
    name='iris_data_set',
    description='registering iris dataset',
    create_new_version=True
)

# Load the dataset into a pandas DataFrame for further processing 
df = dataset.to_pandas_dataframe()

print(df.head(2))

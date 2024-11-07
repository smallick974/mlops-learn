from azureml.core import Workspace, Datastore

# Connect to your Azure ML workspace
ws = Workspace.from_config()

# Register the blob storage account as a datastore
datastore = Datastore.register_azure_blob_container(
    workspace=ws,
    datastore_name='testdata',
    container_name='test-data',
    account_name='irisdata1',
    account_key='S5NPvpt1gsqFITHZi1NcGRC6c7nc02PHFFKnem5b4bLPqiVjpbRuJvq3sVVlMNXeK55xFWea16PL+AStqHVaag==',
    create_if_not_exists=True
)

from azureml.core.dataset import Dataset

# Define the path to the data in the datastore
datastore_path = [(datastore, 'Iris.csv')]

# Create a TabularDataset from the data in the datastore
dataset = Dataset.Tabular.from_delimited_files(path=datastore_path)

# Register the dataset in the workspace (optional)
dataset = dataset.register(
    workspace=ws,
    name='iris_data_set',
    description='Dataset created from local data uploaded to Blob Storage',
    create_new_version=True
)

# Load the dataset into a pandas DataFrame for further processing 
df = dataset.to_pandas_dataframe()

print(df.head(2))

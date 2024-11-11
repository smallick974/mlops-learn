from azureml.core import Workspace, Dataset, Experiment
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import andrews_curves
import os

plt.style.use("fivethirtyeight")
# get_ipython().run_line_magic('matplotlib', 'inline')

# Load the workspace from the saved config file
ws = Workspace.from_config()

# Access the registered dataset
processed_data = Dataset.get_by_name(workspace=ws, name='iris_data_set')

# Load the dataset into a pandas DataFrame
df = processed_data.to_pandas_dataframe()

# Create or get an existing experiment 
experiment = Experiment(workspace=ws, name='exploratory_data_analysis')

# Start a new run 
run = experiment.start_logging()

######################################################################################################################
if not os.path.exists("images"):
    os.mkdir("images")

    image_dir = os.path.abspath("images")
    print(image_dir)
else:
    image_dir = os.path.abspath("images")
    print(image_dir)
######################################################################################################################

plt.figure(figsize=(15,8))
sns.boxplot(x='Species',y='SepalLengthCm',data=df.sort_values('SepalLengthCm',ascending=False))

# Save the plot to a file 
image_path = f"{image_dir}/box_plot.png" 
plt.savefig(image_path) 

# Log the image as an artifact 
run.log_image(name="Box Plot", path=image_path)

######################################################################################################################

df.plot(kind='scatter',x='SepalWidthCm',y='SepalLengthCm')

# Save the plot to a file 
image_path = f"{image_dir}/plot.png" 
plt.savefig(image_path) 

# Log the image as an artifact 
run.log_image(name="Plot", path=image_path)

######################################################################################################################

sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=df, size=5)

# Save the plot to a file 
image_path = f"{image_dir}/joint_plot.png" 
plt.savefig(image_path) 

# Log the image as an artifact 
run.log_image(name="Joint Plot", path=image_path)

######################################################################################################################

sns.pairplot(df, hue="Species", size=3)

# Save the plot to a file 
image_path = f"{image_dir}/pair_plot.png" 
plt.savefig(image_path) 

# Log the image as an artifact 
run.log_image(name="Pair Plot", path=image_path)

######################################################################################################################

df.boxplot(by="Species", figsize=(12, 6))

# Save the plot to a file 
image_path = f"{image_dir}/box_plot_1.png" 
plt.savefig(image_path) 

# Log the image as an artifact 
run.log_image(name="Box Plot 1", path=image_path)

######################################################################################################################

andrews_curves(df, "Species")

# Save the plot to a file 
image_path = f"{image_dir}/andrews_curves.png" 
plt.savefig(image_path) 

# Log the image as an artifact 
run.log_image(name="Andrews Curves", path=image_path)

######################################################################################################################

numeric_df = df.select_dtypes(include=['number'])  # Select numeric columns
correlation_matrix = numeric_df.corr()

# Display the correlation matrix
print(correlation_matrix)

# Save the correlation matrix as a CSV file 
correlation_matrix_path = 'correlation_matrix.csv' 
correlation_matrix.to_csv(correlation_matrix_path, mode='w') 

# Log the CSV file 
run.upload_file(name='outputs/correlation_matrix.csv', path_or_stream=correlation_matrix_path)

######################################################################################################################

plt.subplots(figsize = (8,8))
sns.heatmap(correlation_matrix, annot=True,fmt="f").set_title("Corelation of attributes (petal length,width and sepal length,width) among Iris species")
plt.show()

# Save the plot to a file 
image_path = f"{image_dir}/corr_heat_map.png" 
plt.savefig(image_path) 

# Log the image as an artifact 
run.log_image(name="Corelation matrix heat map", path=image_path)

######################################################################################################################

# Complete the run 
run.complete()


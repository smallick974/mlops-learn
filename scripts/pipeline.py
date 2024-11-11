from azureml.core import Workspace, Experiment, Environment
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep

# Connect to your workspace
ws = Workspace.from_config()

# Define the compute target
compute_target = ws.compute_targets["mllearn-instance"]

# Define the environment
# env = Environment.get(ws, name="Iris-env")

# Create an experiment and submit the pipeline for execution
experiment = Experiment(workspace=ws, name="iris-experiment")

# Start a new run 
run_config = RunConfiguration()

# Define each step in the pipeline
print("defining data upload step")
data_upload_step = PythonScriptStep(name="Data Upload",
                                    script_name="data_upload.py",
                                    compute_target=compute_target,
                                    runconfig=run_config,
                                    source_directory=".")

print("defining data preprocessing step")
data_processing_step = PythonScriptStep(name="Data PreProcessing",
                                        script_name="data_preprocessing.py",
                                        compute_target=compute_target,
                                        runconfig=run_config,
                                        source_directory=".")

print("defining eda step")
eda_step = PythonScriptStep(name="Exploratory Data Analysis",
                            script_name="eda.py",
                            compute_target=compute_target,
                            runconfig=run_config,
                            source_directory=".")

print("defining model training step")
model_training_step = PythonScriptStep(name="Model Training",
                                       script_name="model_training.py",
                                       compute_target=compute_target,
                                       runconfig=run_config,
                                       source_directory=".")

# Define the pipeline by specifying the steps in sequence
print("sequencing pipeling steps")
pipeline_steps = [data_upload_step, data_processing_step, eda_step, model_training_step]
print("creating pipeline")
pipeline = Pipeline(workspace=ws, steps=pipeline_steps)

print("submitting experiment")
pipeline_run = experiment.submit(pipeline)

print("waiting for completion")
pipeline_run.wait_for_completion(show_output=True)

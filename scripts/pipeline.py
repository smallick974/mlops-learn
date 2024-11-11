from azureml.core import Workspace, Experiment, Environment
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep

# Connect to your workspace
ws = Workspace.from_config()

# Define the compute target
compute_target = ws.compute_targets["mllearn-instance"]

# Define the environment
env = Environment.get(ws, name="Iris-env")

# Create an experiment and submit the pipeline for execution
experiment = Experiment(workspace=ws, name="iris-experiment")

# Start a new run 
run_config = experiment.start_logging()

# Define each step in the pipeline

data_upload_step = PythonScriptStep(name="Data Upload",
                                    script_name="data_upload.py",
                                    compute_target=compute_target,
                                    runconfig=run_config,
                                    source_directory=".")

data_processing_step = PythonScriptStep(name="Data PreProcessing",
                                        script_name="data_preprocessing.py",
                                        compute_target=compute_target,
                                        runconfig=run_config,
                                        source_directory=".")

eda_step = PythonScriptStep(name="Exploratory Data Analysis",
                            script_name="eda.py",
                            compute_target=compute_target,
                            runconfig=run_config,
                            source_directory=".")

model_training_step = PythonScriptStep(name="Model Training",
                                       script_name="model_training.py",
                                       compute_target=compute_target,
                                       runconfig=run_config,
                                       source_directory=".")

# Define the pipeline by specifying the steps in sequence
pipeline_steps = [data_upload_step, data_processing_step, eda_step, model_training_step]
pipeline = Pipeline(workspace=ws, steps=pipeline_steps)

pipeline_run = experiment.submit(pipeline)
pipeline_run.wait_for_completion(show_output=True)

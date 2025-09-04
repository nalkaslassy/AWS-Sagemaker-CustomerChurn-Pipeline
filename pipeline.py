import os
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.step_collections import RegisterModel

# Session & role
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Bucket
default_bucket = "bucket name here"

#Parameters
#This is where we upload the csv file to S3
#Make sure to replace with your actual bucket name (S3 bucket names are globally unique)
input_data = ParameterString(
    name="InputData",
    default_value=f"s3://{default_bucket}/data/Telco_customer_churn.csv"
)
instance_type = ParameterString(name="InstanceType", default_value="ml.m5.large")
accuracy_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.85)

# Cache config
#We cache config because  we want to reuse the results of the processing step
#This is useful for large datasets or expensive computations
cache_config = CacheConfig(enable_caching=True, expire_after="30d")

# Processing Step
# Create a SKLearnProcessor for data preprocessing
#It works with scikit-learn and can be used for data preprocessing tasks
sklearn_processor = SKLearnProcessor(
    framework_version="0.23-1",
    role=role,
    instance_type="ml.t3.medium",
    instance_count=1,
    base_job_name="churn-preprocessing"
)


# Define the processing step to preprocess the data
# This step will read the input data, process it, and save the output to S3
# We are telling it to read the data from the input_data parameter at the specified S3 location
# Then it processes the data by executing the script processfeaturestore.py which is specified in the code parameter
# The syntax goes: S3 input data -> ProcessingInput (downloads data to container) -> Script execution (processes data locally) -> ProcessingOutput (uploads results back to S3)
processing_step = ProcessingStep(
    name="ChurnPreprocessing",
    processor=sklearn_processor,
    inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
    outputs=[
        ProcessingOutput(source="/opt/ml/processing/train", destination=f"s3://{default_bucket}/processing/train"),
        ProcessingOutput(source="/opt/ml/processing/validation", destination=f"s3://{default_bucket}/processing/validation"),
        ProcessingOutput(source="/opt/ml/processing/test", destination=f"s3://{default_bucket}/processing/test"),
        ProcessingOutput(source="/opt/ml/processing/model", destination=f"s3://{default_bucket}/processing/model")
    ],
    code="scripts/processfeaturestore.py",
    cache_config=cache_config
)

#Training Step
# Create an XGBoost estimator for training
#This is the model training step where we use the processed data to train a machine learning model
xgb_estimator = Estimator(
    image_uri=sagemaker.image_uris.retrieve("xgboost", sagemaker_session.boto_region_name, version="1.5-1"),
    instance_type="ml.m5.large",
    instance_count=1,
    output_path=f"s3://{default_bucket}/output/model-with-features",
    role=role,
    hyperparameters={
        "objective": "binary:logistic",
        "num_round": 100,
        "eval_metric": "logloss"
    }
)
#Create the training step
#WE read the training and validation data from the processing step outputs
#We use the TrainingInput class to specify the S3 locations of the training and validation data
#in this case S3Uri is the S3 location of the processed data whhich is s3://{default_bucket}/processing/train and s3://{default_bucket}/processing/validation
training_step = TrainingStep(
    name="XGBoostTraining",
    estimator=xgb_estimator,
    inputs={
        "train": TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri,
            content_type="text/csv"
        ),
        "validation": TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[1].S3Output.S3Uri,
            content_type="text/csv"
        )
    },
    cache_config=cache_config
)

#Evaluation Step (updated with ScriptProcessor using XGBoost container)
xgb_image_uri = sagemaker.image_uris.retrieve(
    framework="xgboost",
    region=sagemaker_session.boto_region_name,
    version="1.5-1"
)

evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="metrics.json"
)
#Eval processor
# this is where the eval processor and it does the evaluation of the model
eval_processor = ScriptProcessor(
    image_uri=xgb_image_uri,
    command=["python3"],
    instance_type="ml.t3.medium",
    instance_count=1,
    base_job_name="churn-evaluation",
    role=role,
    sagemaker_session=sagemaker_session
)
#This is where we define the evaluation step where the processor is called and the evluation script is executed
#We use the ScriptProcessor to run the evaluation script
#We specify the inputs and outputs of the evaluation step
#The inputs are the model artifacts from the training step and the test data from the processing step
#The outputs are the evaluation metrics which are saved to S3
#We also specify the code to run which is the evaluate.py script
#The evaluation step will run the script and generate the evaluation metrics
evaluation_step = ProcessingStep(
    name="EvaluateModel",
    processor=eval_processor,
    inputs=[
        ProcessingInput(source=training_step.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/model"),
        ProcessingInput(source=processing_step.properties.ProcessingOutputConfig.Outputs[2].S3Output.S3Uri, destination="/opt/ml/processing/test"),
        ProcessingInput(source=processing_step.properties.ProcessingOutputConfig.Outputs[3].S3Output.S3Uri, destination="/opt/ml/processing/model_features")
    ],
    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation",
            destination=f"s3://{default_bucket}/evaluation"
        )
    ],
    code="scripts/evaluate.py",
    property_files=[evaluation_report],
    cache_config=CacheConfig(enable_caching=False)
)

#Register Model
register_approved = RegisterModel(
    name="RegisterApprovedModel",
    estimator=xgb_estimator,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="ChurnModelPackageGroup",
    approval_status="Approved"
)

register_pending = RegisterModel(
    name="RegisterPendingModel",
    estimator=xgb_estimator,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="ChurnModelPackageGroup",
    approval_status="PendingManualApproval"
)

# Conditional Approval Step
# This step checks if the model accuracy meets the threshold
# If it does, it registers the model as approved; otherwise, it registers it as pending
# We use the evaluation report to get the accuracy metric
condition_step = ConditionStep(
    name="CheckModelAccuracy",
    conditions=[
        ConditionGreaterThanOrEqualTo(
            left=JsonGet(step_name=evaluation_step.name, property_file=evaluation_report, json_path="metrics.accuracy"),
            right=accuracy_threshold
        )
    ],
    if_steps=[register_approved],
    else_steps=[register_pending]
)

# Pipeline Assembly
# Assemble the pipeline with all steps
#You do this by creating a Pipeline object and passing the steps and parameters
pipeline = Pipeline(
    name="ChurnPredictionPipeline",
    parameters=[input_data, instance_type, accuracy_threshold],
    steps=[processing_step, training_step, evaluation_step, condition_step],
    sagemaker_session=sagemaker_session
)

#Execute
if __name__ == "__main__":
    print("Creating or updating pipeline...")
    pipeline.upsert(role_arn=role)
    print("Starting pipeline execution...")
    execution = pipeline.start()
    execution.wait()
    status = execution.describe()["PipelineExecutionStatus"]
    print(f"Pipeline execution complete with status: {status}")

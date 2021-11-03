import os

import mlflow

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://95.216.168.89:19001"
os.environ["MLFLOW_TRACKING_URI"] = "http://95.216.168.89:5000"
os.environ["AWS_ACCESS_KEY_ID"] = "IAM_ACCESS_KEY"
os.environ["AWS_SECRET_ACCESS_KEY"] = "IAM_SECRET_KEY"

MODEL_FLOAT_PATH = "models:/iris_sklearn/production"
MODEL_STRING_PATH = "models:/iris_pyfunc/production"

model_float = mlflow.sklearn.load_model(MODEL_FLOAT_PATH)
model_string = mlflow.pyfunc.load_model(MODEL_STRING_PATH)

import os
import mlflow
import requests

from wrapper import SklearnModelWrapper

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = mlflow.tracking.MlflowClient()


MODEL_PATH = "models:/iris_sklearn/production"
model = mlflow.sklearn.load_model(MODEL_PATH)


result = requests.get("http://waryak:5000/get_iris_thresholds")


wrapped_model = SklearnModelWrapper(
    model=model,
    threshold_0_1=result.json()['threshold_0_1'],
    threshold_1_2=result.json()['threshold_1_2']
)


mlflow.pyfunc.log_model("model", 
                        python_model=wrapped_model, 
                        code_path=["wrapper.py"],
                        registered_model_name="iris_pyfunc")

iris_pyfunc_versions = client.search_model_versions(filter_string="name='iris_pyfunc'")

last_version = iris_pyfunc_versions[-1].version

client.transition_model_version_stage(name="iris_pyfunc", version=last_version, stage="production")

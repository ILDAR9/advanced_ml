import mlflow
import numpy as np

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
    
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        x_array = np.array([model_input])
        prediction = self.model.predict(x_array)
        return prediction
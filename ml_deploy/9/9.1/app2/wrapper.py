import mlflow
import numpy as np


class SklearnModelWrapper(mlflow.pyfunc.PythonModel):

    def __init__(self, model, threshold_0_1, threshold_1_2):
        self.model = model
        self.threshold_0_1 = threshold_0_1
        self.threshold_1_2 = threshold_1_2

    def predict(self, context, model_input):
        x_array = np.array([model_input])
        prediction = self.model.predict(x_array)
        resp_dict = {
            "class": prediction[0], "threshold_0_1": self.threshold_0_1, "threshold_1_2": self.threshold_1_2}
        if prediction < self.threshold_0_1:
            resp_dict["class_str"] = "setosa"
            return resp_dict
        elif prediction < self.threshold_1_2:
            resp_dict["class_str"] = "versicolor"
            return resp_dict
        else:
            resp_dict["class_str"] = "virginica"
            return resp_dict

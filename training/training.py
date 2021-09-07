import mlflow
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()
        self.rmse = None
        self.mse = None
        self.adj_r2 = None

    @staticmethod
    def polynomial_feature(degree, x_train, x_test):
        try:
            poly = PolynomialFeatures(degree)
            x_train = poly.fit_transform(x_train)
            x_test = poly.transform(x_test)
            return x_train, x_test
        except Exception as e:
            print(e)

    def get_model(self):
        try:
            return self.model
        except Exception as e:
            raise e

    def get_rmse(self):
        try:
            return self.rmse
        except Exception as e:
            raise e

    def get_mse(self):
        try:
            return self.mse
        except Exception as e:
            raise e

    def get_adj_r2(self):
        try:
            return self.adj_r2
        except Exception as e:
            raise e

    def calculate_adj_r2(self, x_test, y_test):
        try:
            r2 = self.model.score(x_test, y_test)
            n = x_test.shape[0]
            p = x_test.shape[1]
            self.adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        except Exception as e:
            print(e)

    def mlflow_run(self,
                   x_train,
                   y_train,
                   x_test,
                   y_test,
                   run_name="Linear Regression Model: Air Temperature Model"):
        try:

            remote_server_uri = "http://127.0.0.1:5001"
            mlflow.set_tracking_uri(remote_server_uri)
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.autolog()
                self.model.fit(x_train, y_train)
                self.mse = mean_squared_error(y_test, self.model.predict(x_test))
                self.rmse = np.sqrt(self.mse)
                self.calculate_adj_r2(x_test, y_test)
                mlflow.log_metric("mse", self.mse)
                mlflow.log_metric("rmse", self.rmse)
                mlflow.log_metric("adj r2", self.get_adj_r2())
                run_id = run.info.run_id
            return run_id
        except Exception as e:
            raise e

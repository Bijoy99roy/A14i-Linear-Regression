from training.training import LinearRegressionModel
from mlflow_utils.utils_file import Utils


def mlflow_start():
    try:
        utils_obj = Utils()
        utils_obj.load_data('../final_data.csv')
        x_train, x_test, y_train, y_test = utils_obj.get_train_test_data('Air_temperature')
        x_train, x_test = utils_obj.scale_data(x_train, x_test, 'Process_temperature')

        linear_regression_model = LinearRegressionModel()
        # x_train, x_test = linear_regression_model.polynomial_feature(3, x_train, x_test)
        run_id = linear_regression_model.mlflow_run(
            x_train, y_train, x_test, y_test, 'Linear Regression Model: Air Temperature Model')
        print("Mlflow run_id = {} completed with rmse = {} mse = {} adj r2 = {}".format(
            run_id,
            linear_regression_model.get_rmse(),
            linear_regression_model.get_mse(),
            linear_regression_model.get_adj_r2()))
    except Exception as e:
        print(e)


if __name__ == '__main__':
    mlflow_start()

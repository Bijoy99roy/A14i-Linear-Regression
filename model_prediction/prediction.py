# performing important imports
import os
import numpy as np
import pandas as pd
from application_logging.logger import AppLogger
from prediction_data_validation.prediction_data_validation import PredictionDataValidation
from data_preprocessing.preprocessing import PreProcessing
from file_operation.file_handler import FileHandler


class Prediction:
    def __init__(self):
        self.logger = AppLogger()
        self.file_object = open("prediction_log/prediction_log.txt", 'a+')
        self.pred_data_val = PredictionDataValidation()

    def predict(self):
        """
        This function applies prediction on the provided data
        :return:
        """
        try:
            self.logger.log(self.file_object, 'Start of Prediction', 'Info')
            # initializing PreProcessor object
            preprocessor = PreProcessing(self.file_object, self.logger)
            # initializing FileHandler object
            model = FileHandler(self.file_object, self.logger)
            # getting the data file path
            file = os.listdir('prediction_files/')[0]
            # reading data file
            dataframe = pd.read_csv('prediction_files/'+file)

            # recieving values as tuple
            column_info = self.pred_data_val.get_schema_values()
            data = dataframe.copy()
            data = np.array(preprocessor.scale_data(data, column_info[1]))
            # loading Logistic Regression model
            linear_reg = model.load_model('linear_regressor')
            # predicting
            predicted = linear_reg.predict(data)

            dataframe['predicted'] = predicted
            dataframe.to_csv('prediction_files/Prediction.csv')
            self.logger.log(
                self.file_object,
                'Predction complete!!. Prediction.csv saved in Prediction_File as output. \
                Exiting Predict method of Prediction class ',
                'Info')
            # converting dict array to list
            columninfo = list(column_info[2])
            columninfo.append('prediction')
            return dataframe.to_numpy(), columninfo

        except Exception as e:
            self.logger.log(
                self.file_object,
                'Error occured while running the prediction!! Message: ' + str(e),
                'Error')
            raise e

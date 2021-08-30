# performing important imports
from application_logging.logger import AppLogger
from prediction_data_validation.prediction_data_validation import PredictionDataValidation


class PredictionValidation:
    def __init__(self):
        self.raw_data = PredictionDataValidation()
        self.logger = AppLogger()

    def validation(self):
        f = open("prediction_log/Prediction_Log.txt", "a+")
        try:
            # validating data type
            self.logger.log(f, "Starting data type validation")
            self.raw_data.validate_data_type()
            self.logger.log(f, "data type validation complete!!")
            f.close()

        except Exception as e:
            f.close()
            raise e

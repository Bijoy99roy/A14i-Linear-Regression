import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Utils:
    def __init__(self):
        self.data = None

    def load_data(self, path):
        try:
            self.data = pd.read_csv(path)
            print(self.data.isnull().sum())
        except Exception as e:
            print(e)

    def get_train_test_data(self, target, test_size=0.3, random_state=42):
        try:
            x = self.data.drop([target], axis=1)
            y = self.data[[target]]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
            return x_train, x_test, y_train, y_test
        except Exception as e:
            print(e)

    @staticmethod
    def scale_data(train, test, cols):
        try:
            scaler = StandardScaler()
            train[[cols]] = scaler.fit_transform(train[[cols]])
            test[[cols]] = scaler.transform(test[[cols]])
            return train, test
        except Exception as e:
            print(e)

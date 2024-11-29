import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DataExtractor:
    def __init__(self, default_range = (0, 1)):
        self.range = default_range
        self.scalers = []

    def scale_back(self, scaler_num, data):
        return self.scalers[scaler_num].inverse_transform(data)

    def load_and_scale_data_set(self, file_name, data_columns, predict_column, mean_over=None):
        data = pd.read_csv(file_name, index_col="Date", parse_dates=True)
        scaled = np.array([0] * data.values.shape[0]).T
        whole_data = data_columns.copy()
        whole_data.append(predict_column)
        whole_data = list(set(whole_data))
        columns = []
        for i, column in enumerate(whole_data):
            columns.append(column)
            data[column] = data[column].astype(str).str.replace(',', '').astype(float)
            if mean_over is not None:
                data[column] = data[column].rolling(window=mean_over).mean()
            self.scalers.append(MinMaxScaler(feature_range=self.range))
            c = self.scalers[i].fit_transform(data[column].values.reshape(data.values.shape[0], 1))
            scaled = np.column_stack((scaled, c))

        scaled = np.delete(scaled, 0, 1)
        scaled_df = pd.DataFrame(scaled, columns=columns)
        scaled_df = scaled_df.dropna()
        self.scaled = scaled_df
        return scaled_df

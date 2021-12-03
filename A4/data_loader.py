"""
Created to load the data.

@author: Andy Gabler & Kevin Spike
"""

import pandas as pd

def load_data(file_name="houseSalePrices.csv"):
    frame = pd.read_csv(file_name).set_index("Id")
    data_numpy_array = frame.to_numpy()
    feature_names = frame.columns.to_numpy().astype("str")
    targets = frame.iloc[:,-1:]
    target_names = feature_names[-1:]
    return data_numpy_array[:, :-1], targets.to_numpy().reshape(targets.shape[0]), feature_names[:-1], target_names

if __name__ == "__main__":
    data, targets, feature_names, target_names = load_data()
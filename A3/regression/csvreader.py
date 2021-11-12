"""
Read a csv and return the data in it.

@author: Andy Gabler & Kevin Spike
"""

import numpy as np
import pandas as pd

class CSV_Reader:

    def read_csv(file_name):
        data_frame = pd.read_csv(file_name)
        data_numpy_array = data_frame.to_numpy()
        # For this case, the target is the ERP (last column)
        targets = data_numpy_array[:, 9]
        feature_names = data_frame.columns.to_numpy().astype("str")
        target_names = np.unique(targets)
        data_numpy_array = data_numpy_array[:, 0:9]
        return data_numpy_array, targets, feature_names[:9], target_names, file_name

if __name__ == "__main__":
    in_file_name = input("Enter name of CSV:")
    print("Will attempt to read \"" + in_file_name + "\"...")
    data, targets, feature_names, target_names, file_name = CSV_Reader.read_csv(in_file_name)
    print("Read complete. Variables loaded.")
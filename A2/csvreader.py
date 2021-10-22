"""
Read a csv and return the 

@author: Andy Gabler & Kevin Spike
"""

import pandas as pd

class CSV_Reader:

    def read_csv(file_name):
        data_frame = pd.read_csv(file_name)
        data_numpy_array = data_frame.to_numpy()
        targets = data_frame.index
        feature_names = data_frame.columns.to_numpy().astype("str")
        target_names = targets.unique().to_numpy() # TODO VERIFY THIS IS OKAY
        return data_numpy_array, targets.to_numpy(), feature_names, target_names, file_name

if __name__ == "__main__":
    in_file_name = input("Enter name of CSV:")
    print("Will attempt to read \"" + in_file_name + "\"...")
    data, targets, feature_names, target_names, file_name = CSV_Reader.read_csv(in_file_name)
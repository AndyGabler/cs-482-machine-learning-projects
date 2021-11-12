"""
Common class for pre-processing of the data for Assignment 3.

@author: Andy Gabler & Kevin Spike
"""

import pandas as pd


def load_and_proprocess(file_name):
    # Load the data
    data_frame = pd.read_csv(file_name)
    
    # Set the index as the make-model
    data_frame = data_frame.set_index(keys=[data_frame.columns[0], data_frame.columns[1]])
    
    """
    MMIN, MMAX and PRP have high correlation between each other. 
    Only keep MMAX
    """
    data_frame = data_frame.drop(["PRP", "MMIN"], axis=1)
    
    # Separate target from data
    target = data_frame["ERP"].to_numpy()
    data_frame = data_frame.drop(["ERP"], axis=1)
    return data_frame, target

def frame_to_numpy(frame):
    data_array = frame.to_numpy()
    feature_names = frame.columns.to_numpy().astype("str")
    description_labels = frame.index.to_series().to_numpy()
    
    return data_array, feature_names, description_labels
    
if __name__ == "__main__":
    in_file_name = input("Enter name of Machine Data CSV:")
    print("Will attempt to process \"" + in_file_name + "\"...")
    data, targets = load_and_proprocess(in_file_name)
    array_form, feature_names, description_labels = frame_to_numpy(data)
    print("Read complete. Variables loaded.")
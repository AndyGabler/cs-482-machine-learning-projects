"""
Created for the "Meet The Data Section" of Assignment 3. Gathers the
required meta-data from the Haberman data. 

@author: Andy Gabler & Kevin Spike
"""
from csvreader import CSV_Reader as csv
import pandas as pd
import numpy as np

data, targets, feature_names, target_names, file_name = csv.read_csv("haberman.data")

print("Number of Features: " + str(feature_names.shape[0]))
print("Description of Features:\n{}".format(feature_names))
print("Description of Target: {}".format(target_names))
print("Number of Samples: " + str(data.shape[0]))
print("Description of data: Haberman data of size", str(data.shape[0]), "from file", file_name)
print("First Five Rows of Data:\n{}".format(data[0:5]))

"""
START OF THE CORRELATION SECTION
"""
# Start by creating a dataframe, just going to reread CSV to avoid casting
data_frame = pd.read_csv("haberman.data")
# First two columns are make-model, aka indices. we don't need these
correlation_frame = data_frame.corr()
print("Correlation of Features:\n{}".format(correlation_frame))
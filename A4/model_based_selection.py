"""
Model based feature selection per Task 6.

@author: Andy Gabler & Kevin Spike
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from transformation import data_transformation

def model_feature_select(do_plot=False):
    data, targets, feature_names = data_transformation()
    
    select = SelectFromModel(
        LinearRegression(),
        threshold=-np.inf, # Only select on max_features
        max_features=int(feature_names.shape[0] / 2) # Throttle at half features
    )
    select.fit(data, targets)
    data_transform = select.transform(data)
    feature_transform = feature_names[select.get_support()]
    
    # Do plots if necesary
    if do_plot:
        fig0, ax0 = plt.subplots()
        ax0.matshow(select.get_support().reshape(1, -1), cmap='gray_r')
        plt.xlabel("Feature Index")
    
    return data_transform, targets, feature_transform

if __name__ == "__main__":
    data, targets, feature_names = model_feature_select(True)
    print("Remaining Features:\n", feature_names)
"""
Univariate feature pruning per Task 4.

@author: Andy Gabler & Kevin Spike
"""

from information_fill import do_information_fill
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression

def univariate_selection(do_plot=False):
    data, targets, feature_names, target_names = do_information_fill()

    # We're cutting out 30%, meaning retain 70%
    select = SelectPercentile(percentile=70, score_func=f_regression)
    select.fit(data, targets)
    data = select.transform(data)
    
    # Do plots if necesary
    if do_plot:
        fig0, ax0 = plt.subplots()
        ax0.matshow(select.get_support().reshape(1, -1), cmap='gray_r')
        plt.xlabel("Feature Index")
        
        fig1, ax1 = plt.subplots()
        ax1.bar(np.arange(0, select.scores_.shape[0], 1), select.pvalues_)
        plt.xlabel("Feature Index")
        plt.ylabel("Correlation Value")
    
    # This can be get_feature_names_out
    feature_names = feature_names[select.get_support()]
    
    return data, targets, feature_names

if __name__ == "__main__":
    data, targets, feature_names = univariate_selection(True)
    print("Features Retained:\n", feature_names)
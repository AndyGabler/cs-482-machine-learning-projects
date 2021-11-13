"""
Created for the ROC Curve task in Assignment 3.

@author: Andy Gabler & Kevin Spike
"""

from csvreader import CSV_Reader as csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Load and split the data
data, targets, feature_names, target_names, file_name = csv.read_csv("haberman.data")
X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=0)

# Train up our model
# TODO conider using L1 penalty?
logisticEstimator = LogisticRegression().fit(X_train, y_train)

false_positive_rate, true_positive_rate, thresholds = roc_curve(
    y_test, 
    logisticEstimator.decision_function(X_test),
    pos_label=target_names[1]
)
# The result cannot be right...
plt.plot(false_positive_rate, true_positive_rate, label="ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
close_zero = np.argmin(np.abs(thresholds))
plt.plot(false_positive_rate[close_zero], true_positive_rate[close_zero], 'o', markersize=10, label="threshold zero", fillstyle="none", c='k', mew=2)
plt.legend(loc=4)

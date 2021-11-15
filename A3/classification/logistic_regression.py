"""
Created for Logistic Regression (No Cross Validation) for Assignment 3.
Combines the sections where we calculate accuracy, precision, recall, and
sensitivity.

@author: Andy Gabler & Kevin Spike
"""

from csvreader import CSV_Reader as csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load and split the data
data, targets, feature_names, target_names, file_name = csv.read_csv("haberman.data")
X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=0)

# Train up our model, using default L2 since L1 produced less desireable results
logisticEstimator = LogisticRegression().fit(X_train, y_train)

# Now use a confusion matrix to get TN, TP, FN, FP
confusion = confusion_matrix(
    y_test, logisticEstimator.predict(X_test),
    labels=[target_names[1], target_names[0]] #(switch survived to true class)
)

true_negative = confusion[0, 0] #TN
false_positive = confusion[0, 1] #FP
false_negative = confusion[1, 0] #FN
true_positive = confusion[1, 1] #TP

print("Recall/Sensitivity: {}".format(true_positive/(true_positive + false_negative)))
print("Precision: {}".format(true_positive/(true_positive + false_positive)))
print("Accuracy: {}".format((true_positive + true_negative)/(true_positive + false_negative + true_negative + false_positive)))
print("Specificity: {}".format(true_negative/(true_negative + false_positive)))

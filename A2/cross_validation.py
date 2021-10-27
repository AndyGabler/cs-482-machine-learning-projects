"""
Created for the "k-NN With Cross Validation Section"

@author: Andy Gabler & Kevin Spike
"""

import pandas as pd
import numpy as np
from csvreader import CSV_Reader as csv
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

# Load the data
data, targets, feature_names, target_names, file_name = csv.read_csv("wine.data")

# Set up the k-fold and instantiate initial knn model
kFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
knnModel = KNeighborsClassifier(n_neighbors=5)

# Accuracy array for later
training_accuracies = []
testing_accuracies = []

# For indexes returned by the k-fold, do the scoring
for train_index, test_index in kFold.split(data, targets):
    # Use the indexes to get folds of the data
    training_data_fold = data[train_index]
    training_target_fold = targets[train_index]
    testing_data_fold = data[test_index]
    testing_target_fold = targets[test_index]
    
    # Fit model to current fold
    knnModel.fit(training_data_fold, training_target_fold)
    
    # Score the model based on the current fold
    training_score = knnModel.score(training_data_fold, training_target_fold)
    testing_score = knnModel.score(testing_data_fold, testing_target_fold)
    
    # Save the scores for later
    training_accuracies.append(training_score)
    testing_accuracies.append(testing_score)

# Get a mean of the accuracies
training_accuracy_mean = np.array(training_accuracies).mean()
testing_accuracy_mean = np.array(testing_accuracies).mean()

# Put mean in the accuracies
training_accuracies.append(training_accuracy_mean)
testing_accuracies.append(testing_accuracy_mean)

# Create a Pandas dataframe of the accuracies and print it
accuracy_frame = pd.DataFrame({
    "Fold-1" : [training_accuracies[0], testing_accuracies[0]],
    "Fold-2" : [training_accuracies[1], testing_accuracies[1]],
    "Fold-3" : [training_accuracies[2], testing_accuracies[2]],
    "Fold-4" : [training_accuracies[3], testing_accuracies[3]],
    "Fold-5" : [training_accuracies[4], testing_accuracies[4]],
    "Mean" : [training_accuracies[5], testing_accuracies[5]]
})
accuracy_frame = accuracy_frame.rename(index={0 : "Training Accuracy", 1 : "Test Accuracy"})
print(accuracy_frame)
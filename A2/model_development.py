"""
Created for the "Model Development and Training" of Assignment 2.

@author: Andy Gabler & Kevin Spike
"""
import matplotlib.pyplot as plt
from csvreader import CSV_Reader as csv
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load up the actual data
data, targets, feature_names, target_names, file_name = csv.read_csv("wine.data")

# Calculate the max number of neighbors as sqrt(n) + 3
max_neighbor_numbers = int(math.ceil(math.sqrt(data.shape[0]))) + 3
# Perform the actual split
training_data, test_data, training_target, test_target = train_test_split(data, targets, test_size=0.2, random_state=0)

# Meta-data for a maximum finding algorithm
best_score = -1.0
best_knn_model = None
best_neighbor_count = 0
neighbor_count = 1

# Training and testing scores for printout later
training_scores = []
scores = []

# While we have more neighbor counts to try, try them
while neighbor_count <= max_neighbor_numbers:
    # Set up a KNN model and train it, with given neighbor count
    knn = KNeighborsClassifier(n_neighbors=neighbor_count)
    knn.fit(training_data, training_target)    
    
    # Score the model
    kscore = knn.score(test_data, test_target)
    
    training_scores.append(knn.score(training_data, training_target))
    scores.append(kscore)
    
    # Check if current score is the new best score
    if kscore > best_score:
        best_score = kscore
        best_knn_model = knn
        best_neighbor_count = neighbor_count
    
    neighbor_count = neighbor_count + 1

# Make the data plotable
training_scores = np.array(training_scores)
scores = np.array(scores)
neighbor_counts = np.arange(1, max_neighbor_numbers + 1, 1)

# Accuracy plot
plt.plot(neighbor_counts, training_scores)
plt.plot(neighbor_counts, scores, '--')
plt.title("Model Scores")
plt.xlabel("Neighbors")
plt.ylabel("Accuracy")
plt.legend(["Training Data", "Test Data"])
plt.show()

print("\nBest KNN Score Was With {} Neighbors and Score Was {}".format(best_neighbor_count, best_score))
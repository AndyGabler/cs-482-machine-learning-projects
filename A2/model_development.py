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

data, targets, feature_names, target_names, file_name = csv.read_csv("wine.data")
max_neighbor_numbers = int(math.ceil(math.sqrt(data.shape[0]))) + 3
training_data, test_data, training_target, test_target = train_test_split(data, targets, random_state=0)

best_score = -1.0
best_knn_model = None
best_neighbor_count = 0
neighbor_count = 1

training_scores = []
scores = []

while neighbor_count <= max_neighbor_numbers:
    knn = KNeighborsClassifier(n_neighbors=neighbor_count)
    knn.fit(training_data, training_target)    
    kscore = knn.score(test_data, test_target)
    
    training_scores.append(knn.score(training_data, training_target))
    scores.append(kscore)
    
    if kscore > best_score:
        best_score = kscore
        best_knn_model = knn
        best_neighbor_count = neighbor_count
    
    neighbor_count = neighbor_count + 1

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
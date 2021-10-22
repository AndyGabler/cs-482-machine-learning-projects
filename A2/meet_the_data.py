"""
Created for the "Meet The Data Section" of Assignment 2. Gathers the
required meta-data from the Wine data.

@author: Andy Gabler & Kevin Spike
"""
import matplotlib.pyplot as plt
from csvreader import CSV_Reader as csv
import numpy as np

data, targets, feature_names, target_names, file_name = csv.read_csv("wine.data")

print("Number of Features: " + str(feature_names.shape[0]))
print("Number of Samples: " + str(data.shape[0]))
print("Description of Features:\n{}".format(feature_names))
print("Description of Target: {}".format(target_names))
print("First Five Rows of Data:\n{}".format(data[0:5]))

# As our most influential features, we chose Alcohol and Flavinoids
featureIndex0 = 0
featureIndex1 = 6

"""
MAKING THE HISTORGRAMS
"""
fig0, ax0 = plt.subplots()
ax0.hist(data[:, featureIndex0])
plt.ylabel("Frequency")
plt.xlabel(feature_names[featureIndex0])

fig1, ax1 = plt.subplots()
ax1.hist(data[:, featureIndex1])
plt.ylabel("Frequency")
plt.xlabel(feature_names[featureIndex1])

"""
MAKING THE SCATTER PLOTS
"""
fig2, ax2 = plt.subplots()
class1Indices = np.where(targets == 1)[0]
class2Indices = np.where(targets == 2)[0]
class3Indices = np.where(targets == 3)[0]

ax2.scatter(
    data[class1Indices, featureIndex0], 
    data[class1Indices, featureIndex1], 
    marker = 'o',
    label="Class 1"
)
ax2.scatter(
    data[class2Indices, featureIndex0], 
    data[class2Indices, featureIndex1], 
    marker = 'o', 
    label="Class 2"
)
ax2.scatter(
    data[class3Indices, featureIndex0], 
    data[class3Indices, featureIndex1],  
    marker = 'o',
    label="Class 3"
)
ax2.legend(numpoints=1)
plt.xlabel(feature_names[featureIndex0])
plt.ylabel(feature_names[featureIndex1])

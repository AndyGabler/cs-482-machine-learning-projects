"""
Meet the data section.

@author: Andy Gabler & Kevin Spike
"""

from data_loader import load_data
import matplotlib.pyplot as plt
import numpy as np
import mglearn

data, targets, feature_names, target_names = load_data()

print("Number of Features: " + str(feature_names.shape[0]))
print("Description of Features:\n{}".format(feature_names))
print("Description of Target: {}".format(target_names))
print("Number of Samples: " + str(data.shape[0]))
print("First Five Rows of Data:\n{}".format(data[0:5]))

# Then do histograms
fig, axes = plt.subplots(int(feature_names.shape[0] / 3), 3, figsize=(10, 20))
ax = axes.ravel()

lt_50k = data[targets < 50000]
lt_100k = data[(100000 > targets) * (targets >= 50000)]
lt_200k = data[(200000 > targets) * (targets >= 100000)]
lt_300k = data[(300000 > targets) * (targets >= 200000)]
lt_400k = data[(400000 > targets) * (targets >= 300000)]
lt_500k = data[(500000 > targets) * (targets >= 400000)]
gte_500k = data[targets >= 500000]


for i in range(feature_names.shape[0]):
    # Wrap in try-catch since only numeric columns can be represented.
    try:
        _, bins = np.histogram(data[:, i], bins=50)
        ax[i].hist(lt_50k[:, i], bins=bins, color=mglearn.cm3(0), alpha=.5)
        ax[i].hist(lt_100k[:, i], bins=bins, color=mglearn.cm3(2), alpha=.5)
        ax[i].hist(lt_200k[:, i], bins=bins, color=mglearn.cm3(4), alpha=.5)
        ax[i].hist(lt_300k[:, i], bins=bins, color=mglearn.cm3(6), alpha=.5)
        ax[i].hist(lt_400k[:, i], bins=bins, color=mglearn.cm3(8), alpha=.5)
        ax[i].hist(lt_500k[:, i], bins=bins, color=mglearn.cm3(10), alpha=.5)
        ax[i].hist(gte_500k[:, i], bins=bins, color=mglearn.cm3(12), alpha=.5)
        ax[i].set_title(feature_names[i])
        ax[i].set_yticks(())
    except:
        continue

ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["<$50k", "<$100k", "<$200k", "<$300k", "<$400k", "<$500k", "$500k>"], loc="best")
fig.tight_layout()

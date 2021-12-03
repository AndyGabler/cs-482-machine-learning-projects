"""
Principle Component Analysis per Task 7.

@author: Andy Gabler & Kevin Spike
"""

from model_based_selection import model_feature_select
from sklearn.decomposition import PCA
import math

def get_pca_components():
    data, targets, feature_names = model_feature_select()
    
    # At this point, everything is a OHE feature or previously scaled.
    pca = PCA(n_components=round(feature_names.shape[0] / 10))
    
    pca.fit(data)
    pca_data = pca.transform(data)
    
    return pca_data, targets

if __name__ == "__main__":
    data, targets = get_pca_components()
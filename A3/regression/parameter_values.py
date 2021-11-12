"""
Created for the Parameter Values of Assignment 3.

@author: Andy Gabler & Kevin Spike
"""

from scoring import develop_models
from parameter_tuning import ridge_lasso_alpha_parameters
import matplotlib.pyplot as plt
import numpy as np

# We trust that this method gets us the correct alpha values.
models = develop_models()
lasso_model = models[0]
ridge_model = models[1]
linear_model = models[2]

# TODO, technically correct but could use some prettying/tuning
fig, ax = plt.subplots()
ax.scatter(
    np.arange(0, lasso_model.coef_.shape[0], 1), 
    lasso_model.coef_, 
    marker = 'o',
    label="Lasso"
)
ax.scatter(
    np.arange(0, ridge_model.coef_.shape[0], 1), 
    ridge_model.coef_, 
    marker = 'd',
    label="Ridge"
)
ax.scatter(
    np.arange(0, linear_model.coef_.shape[0], 1), 
    linear_model.coef_, 
    marker = '*',
    label="Linear"
)

ax.legend(numpoints=1)
plt.xlabel("coefficient index")
plt.ylabel("coefficient magnitude")


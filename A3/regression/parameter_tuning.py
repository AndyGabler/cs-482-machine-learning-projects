"""
Created for the "Parameter Tuning" of Assignment 3.

@author: Andy Gabler & Kevin Spike
"""

from preprocess import frame_to_numpy
from preprocess import load_and_proprocess
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def ridge_lasso_alpha_parameters(data, targets):
    
    # zero is technically the best, but excluding because the model does not like it
    param_grid = {'alpha': [0.1, 1.0, 10.0, 20.0, 50.0, 100.0]}

    X_train, X_test, y_train, y_test = train_test_split(data, targets, random_state=0)

    ridge_grid_search = GridSearchCV(Ridge(), param_grid, cv=5)
    lasso_grid_search = GridSearchCV(Lasso(), param_grid, cv=5)

    ridge_grid_search.fit(X_train, y_train)
    lasso_grid_search.fit(X_train, y_train)
    
    return ridge_grid_search.best_params_['alpha'], lasso_grid_search.best_params_['alpha']

if __name__ == "__main__":
    data_frame, targets = load_and_proprocess("machine.data")
    data, features, description_labels = frame_to_numpy(data_frame)
    ridge_alpha, lasso_alpha = ridge_lasso_alpha_parameters(data, targets)
    print("Ridge Best Alpha: {}".format(ridge_alpha))
    print("Lasso Best Alpha: {}".format(lasso_alpha))
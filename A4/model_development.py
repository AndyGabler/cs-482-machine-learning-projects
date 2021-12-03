"""
Model development per Task 8.

@author: Andy Gabler & Kevin Spike
"""
# Start Time 1:58 pm
from pca import get_pca_components
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

data, targets = get_pca_components()

param_grid = {
    'kernel' : ['rbf', 'linear', 'sigmoid', 'poly'], 
    'degree' : [1, 2, 3, 4],
    'C': [0.001, 0.01, 0.1, 1, 10, 100], 
    'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
    'epsilon': [0.001, 0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(SVR(), param_grid, cv=5)
X_train, X_test, y_train, y_test = train_test_split(
    data, targets, random_state=0
)

grid_search.fit(X_train, y_train)
print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test)))
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
print("Best estimator:\n{}".format(grid_search.best_estimator_))

parameter_results = grid_search.cv_results_

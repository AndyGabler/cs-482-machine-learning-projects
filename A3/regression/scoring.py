"""
Created for the "Scoring" of Assignment 3.

@author: Andy Gabler & Kevin Spike
"""

from parameter_tuning import ridge_lasso_alpha_parameters
from preprocess import frame_to_numpy
from preprocess import load_and_proprocess
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import math
import pandas as pd

def develop_models():
    data_frame, targets = load_and_proprocess("machine.data")
    data, features, description_labels = frame_to_numpy(data_frame)
    X_train, X_test, y_train, y_test = train_test_split(data, targets, random_state=0)

    ridge_alpha, lasso_alpha = ridge_lasso_alpha_parameters(data, targets)

    # Instantiate models
    lasso_model = Lasso(alpha=lasso_alpha)
    ridge_model = Ridge(alpha=ridge_alpha)
    linear_model = LinearRegression()

    # Fit models
    lasso_model.fit(X_train, y_train)
    ridge_model.fit(X_train, y_train)
    linear_model.fit(X_train, y_train)
    return lasso_model, ridge_model, linear_model, X_test, y_test


if __name__ == "__main__":
    lasso_model, ridge_model, linear_model, X_test, y_test = develop_models()
    # Recall, simply calling .score returns the R squared value
    lasso_r_square = lasso_model.score(X_test, y_test)
    ridge_r_square = ridge_model.score(X_test, y_test)
    linear_r_square = linear_model.score(X_test, y_test)

    # Take square root of the mean_squared_error (MSE) function for RMSE
    lasso_rmse = math.sqrt(mean_squared_error(y_test, lasso_model.predict(X_test)))
    ridge_rmse = math.sqrt(mean_squared_error(y_test, ridge_model.predict(X_test)))
    linear_rmse = math.sqrt(mean_squared_error(y_test, linear_model.predict(X_test)))

    # Finally, put it in a DataFrame
    score_frame = pd.DataFrame({
        "R^2" : {
            "Lasso Regression" : lasso_r_square, 
            "Ridge Regression" : ridge_r_square, 
            "Linear Regression" : linear_r_square
        },
        "RMSE" : {
            "Lasso Regression" : lasso_rmse, 
            "Ridge Regression" : ridge_rmse, 
            "Linear Regression" : linear_rmse
        }
    })

    print(score_frame)
"""
Univariate feature pruning per Task 4.

@author: Andy Gabler & Kevin Spike
"""

from information_fill import do_information_fill
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression

def univariate_selection():
    data, targets, feature_names, target_names = do_information_fill()

    # We're cutting out 30%, meaning retain 70%
    select = SelectPercentile(percentile=70, score_func=f_regression)
    select.fit(data, targets)
    data = select.transform(data)

    # This can be get_feature_names_out
    feature_names = feature_names[select.get_support()]
    return data, targets, feature_names

if __name__ == "__main__":
    data, targets, feature_names = univariate_selection()
"""
Transformation of features per Task 5.

@author: Andy Gabler & Kevin Spike
"""

from sklearn.preprocessing import StandardScaler
from univariate_feature_selection import univariate_selection

def standard_scale(to_scale):
    scaler = StandardScaler()
    scaler.fit(to_scale.reshape((to_scale.shape[0], 1)))
    return scaler.transform(to_scale.reshape((to_scale.shape[0], 1))).reshape((to_scale.shape[0]))

def data_transformation():
    """
    Features considered:
        6   LotFrontage
        7   LotArea
        40  YearBuilt
        41  YearRemodAdd
        70  MasVnrArea
        104 BsmtFinSF1
        109 BsmtUnfSf
        110 TotalBsmtSf
        124 1stFlrSF
        125 2ndFlrSF
        126 GrLivArea
        152 GarageYrBlt
        158 GarageArea
        171 WoodDeckSF
        172 OpenPorchSF
        173 EnclosedPorch
        174 3SsnPorch
        175 ScreenPorch
        176 PoolArea
    Features where 0 has a special value were skipped
    """
    data, targets, feature_names = univariate_selection()
    
    # One-Hot-Encoding has already been done at this point.
    
    """
    For the LotArea (index 7) always has values. Good to know where it stacks
    up. Using a Standard Scalar.
    """
    data[:, 7] = standard_scale(data[:, 7])
    
    """
    For the 1stFlrSF (index 124) seems important. Good to know how far
    off from standard deviation.
    """
    data[:, 124] = standard_scale(data[:, 124])
    
    """
    For the YearBuilt (index 40) and YearRemodAdd (index 41). All have
    values and seems relevant to how updated the house is.
    """
    data[:, 40] = standard_scale(data[:, 40])
    data[:, 41] = standard_scale(data[:, 41])
    
    return data, targets, feature_names

if __name__ == "__main__":
    data, targets, feature_names = data_transformation()
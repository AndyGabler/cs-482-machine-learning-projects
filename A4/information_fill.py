"""
Fill in missing information per Task 3.

@author: Andy Gabler & Kevin Spike
"""

from data_loader import load_data
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def check_numeric(candidate):
    try:
        float(candidate)
        return True
    except ValueError:
        return False

def eager_one_hot_encode(data, feature_names):
    # Do one hot encoding on all string columns
    
    new_data = np.zeros((data.shape[0], 1))
    new_feature_names = np.array([]).astype("str")
    
    for i in range(feature_names.shape[0]):
        #if not check_numeric(data[0, i]) data[:, i]:
        if not np.vectorize(check_numeric)(data[:, i]).all():
            encoder = OneHotEncoder(sparse=False)
            existing_column = data[:, i]
            new_columns = encoder.fit_transform(existing_column.reshape((existing_column.shape[0], 1)))
            new_features = encoder.get_feature_names([feature_names[i]])
            
            new_data = np.hstack((new_data, new_columns))
            new_feature_names = np.concatenate((new_feature_names, new_features))
        else:
            new_data = np.hstack((new_data, data[:, i].reshape((data.shape[0], 1))))
            new_feature_names = np.concatenate((new_feature_names, np.array([feature_names[i]])))
    
    return new_data[:, 1:], new_feature_names

def do_information_fill():
    
    data, targets, feature_names, target_names = load_data()
    
    data, feature_names = eager_one_hot_encode(data, feature_names)
    
    # 7rd column (LotFrontage) has missing values.
    # LotFrontage is distance from property to road.
    # Replacing with zeroes make sense here.
    data[:, 6] = np.nan_to_num(data[:, 6].astype("float"))
    
    # 220th column (GarageYrBlt) has missing values.
    # No value means no garage, give it the same treatment as LotFrontage
    data[:, 219] = np.nan_to_num(data[:, 219].astype("float"))
    
    # 116th column (MasVnrArea) has missing values.
    # Presumably, this is for those who have MasVnrType of None. Fill with 0.
    data[:, 115] = np.nan_to_num(data[:, 115].astype("float"))

    # In OHE, Some of the Nones interpreted as nan for MasVnrType.
    # MasVnrType_None (113th) and MasVnrType_nan (115th) ought to be
    # combined to one single column.
    nan_mas_vnr_type_indices = data[:, 114] == 1.0
    # Copy over the true values...
    data[nan_mas_vnr_type_indices, 112] = data[nan_mas_vnr_type_indices, 114]
    # ... and cut the redundant column out of the picture
    data = np.hstack((data[:, :114], data[:, 115:]))
    feature_names = np.concatenate((feature_names[:114], feature_names[115:]))
    
    return data, targets, feature_names, target_names
    
if __name__ == "__main__":
    data, targets, feature_names, target_names = do_information_fill()
    #a = np.argwhere(np.isnan(data.astype("float"))) #PLACEHOLDER TO HUNT DOWN NANs
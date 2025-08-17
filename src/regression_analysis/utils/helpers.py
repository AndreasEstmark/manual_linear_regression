import numpy as np


def check_for_outliers(X: np.ndarray):
    """
    Method to check for outliers in data
    """

    pass


def log_transform_dependent_variable(y:np.ndarray) -> np.ndarray:
    """
    Method to log transform the dependent variable.
    This is useful for linear regression models when the dependent variable is skewed.
    """
    if not isinstance(y, np.ndarray):
        raise TypeError("y must be a numpy array.")
    
    return np.log(y + 1)  # Adding 1 to avoid log(0)


def impute_missing_values(X:np.ndarray, imputation_method: str = "mean")->np.ndarray:
    """ 
    Method to impute missing values in the data.
    """
    pass


def standardize_data(X: np.ndarray) -> np.ndarray:
    """
    Method to standardize the data.
    This is useful for linear regression models when the features are on different scales.
    """
    pass

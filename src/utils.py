import numpy as np

"""
Utility functions for linear regression model diagnostics and checks.
"""

def compute_double_sided_t_test_for_coefficients(X: np.matrix, y: np.array, beta: np.array):
    """
    Method to compute the t-statistic for the coefficients of the linear regression model.
    H0: The coefficient is equal to zero (beta coefficient)
    H1: The coefficient is not equal to zero (beta coefficient)
    Test statistic: The test uses a t-statistic calculated from the sample data, following a 
    a t-distribution with n - p degrees of freedom, where n is the number of observations and p is the number of parameters.
    Significance level: The significance level is typically set at 0.05, which corresponds to a 95% confidence level.
    Returns: A dictionary with the t-statistic and p-value for each coefficient.
    """
    pass


def compute_f_statistic(X: np.matrix, y: np.array, beta: np.array):
    """
    Method to compute the F-statistic for the linear regression model.
    
    """
    pass

def check_multicollinearity_in_regressors(X: np.matrix):
    """
    Method to check for multicollinearity in the linear regression model.
    """
    pass

def check_if_matrix_is_invertible(X: np.matrix):
    """
    Method to check if the matrix is invertible. Will maybe calculcate manually, using numpy for now.
    Raises a ValueError if the matrix is not invertible.
    """
    if np.linalg.det(X) == 0:
        raise ValueError("The matrix is not invertible. Check your data.")
    return True


def check_for_very_skewed_regressors(X: np.matrix):
    """
    Method to check for very skewed regressors in the linear regression model.
    """
    pass

def check_for_outliers_in_regressors(X: np.matrix):
    """
    Method to check for outliers in the regressors of the linear regression model. Probably use IQR in the beginning.
    """
    pass

def check_if_constant_term_is_present(X: np.matrix):
    """
    Method to check if the constant term is present in the linear regression model.
    If not, it will add a column of ones to the matrix X.
    """
    pass

class GaussMarkovAssumptions:

    def compute_homoscedasticity(X: np.matrix, y: np.array, beta: np.array):
        """
        Method to compute the homoscedasticity of the linear regression model.
        """
        pass
    
    def check_linearity(X: np.matrix, y: np.array):
        """
        Method to check the linearity of the linear regression model.
        """
        pass


import numpy as np



"""
Utility functions for linear regression model diagnostics and checks.
"""

def compute_double_sided_t_test_for_coefficients(X: np.ndarray, y: np.ndarray, beta: np.ndarray, alpha: float = 0.05):
    """
    Method to compute the t-statistic for the coefficients of the linear regression model.
    H0: The coefficient is equal to zero (beta coefficient)
    H1: The coefficient is not equal to zero (beta coefficient)
    Test statistic: The test uses a t-statistic calculated from the sample data, following a 
    a t-distribution with n - p degrees of freedom, where n is the number of observations and p is the number of parameters.
    Significance level: The significance level is typically set at 0.05, which corresponds to a 95% confidence level.
    Returns: A dictionary with the t-statistic and p-value for each coefficient.
    """

    y = np.asarray(y).reshape(-1, 1)       # (n, 1)
    beta = np.asarray(beta).reshape(-1, 1) # (p, 1)

    SSE: np.array = y.T @ y - beta.T @X.T@y  # scalar

    # TODO 
    # s_squared = SSE / 

    print(f"SSE: {SSE}")

def compute_f_statistic(X: np.ndarray, y: np.ndarray, beta: np.ndarray):
    """
    Method to compute the F-statistic for the linear regression model.
    
    """
    pass

def check_multicollinearity_in_regressors(X: np.ndarray) -> np.ndarray:
    """
    Method to check for multicollinearity in the linear regression model. Using VIF
    X is the full design matrix.
    A value of VIF > 10 is considered high and indicates multicollinearity. 
    The function only does for one regressor now. 
    """

    # TODO implement loop or some list comprehension.  
    from regression_analysis.models.ols import OLSRegression

    k_shape = X.shape[1]

    list_of_vifs = []

    for i in range(X.shape[1]):
        # For each regressor x_i, we compute the VIF
        x_i = X[:, i]
        mask = np.arange(k_shape) != i
        x_noti = X[:, mask]

        x_noti = np.column_stack((np.ones(len(x_i)), x_noti))

        model = OLSRegression().simple_fit(x_noti, x_i)

        list_of_vifs.append(1 / (1 - model.r_squared))
    

    return list_of_vifs

def check_if_matrix_is_invertible(X: np.ndarray):
    """
    Method to check if the matrix is invertible. Will maybe calculcate manually, using numpy for now.
    Raises a ValueError if the matrix is not invertible.
    """
    if np.linalg.det(X) == 0:
        raise ValueError("The matrix is not invertible. Check your data.")
    return True


def check_for_very_skewed_data(X: np.ndarray):
    """
    Method to check for very data
    """
    pass

def check_for_outliers_in_regressors(X: np.ndarray):
    """
    Method to check for outliers in data
    """
    pass

def check_if_constant_term_is_present(X: np.ndarray):
    """
    Method to check if the constant term is present in the linear regression model.
    If not, it will add a column of ones to the matrix X.
    """
    pass

def compute_homoscedasticity(X: np.ndarray, y: np.array, beta: np.array):
    """
    Method to compute the homoscedasticity of the linear regression model.
    """
    pass

def check_linearity(X: np.ndarray, y: np.array):
    """
    Method to check the linearity of the linear regression model.
    """
    pass

def analyze_residuals_of_model(X: np.ndarray, y:np.array,beta: np.array):
    """
    Method to analyze the residuals of the  regression model.
    """
    pass


def calculate_type_1_error():
    """
    This is the risk of false alarm
    """
    pass

def calculate_type_2_error():
    """
    This is the risk of missed detection
    """
    pass
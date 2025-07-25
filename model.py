import pandas as pd
from numpy.linalg import inv



# first define the model:

class LinearRegression:
    """Linear Regression Model Class
    This class implements a simple linear regression model."""

    def __init__(self):

        self.coef_ = None
        self.intercept_ = None

    def __str__(self):
        return f"<LinearRegression: intercept={self.intercept_}, R²={self.r_squared_}>"

    def fit (X: np.matrix, y: np.array):
        """
        Method to compute the parameters of linear regression and R squared.
        """
        # just to check they are in correct format 
        # X is a matrix of shape (n_samples, n_features)
        # y is a vector of shape (n_samples,)
        if not isinstance(X, np.matrix):
            raise ValueError("X must be a numpy matrix.")
        
        if not isinstance(y, np.ndarray):
            raise ValueError("y must be a numpy array.")
        

        # need to rewrite:
        beta = inv(np.transpose(x).dot(x)).dot(np.transpose(x)).dot(y)
        e = y - x.dot(beta)
        squared_error = (np.transpose(e).dot(e))
        y_demeaned_b = np.transpose(y - np.mean(y)).dot(y - np.mean(y))
        r2 = 1 - (np.transpose(e).dot(e) / y_demeaned_b)

        return beta, r2
    

    def predict(self, X: np.matrix):
        pass


    def r_squared(self, X: np.matrix, y: np.array):
        pass

    def p_values_for_coefficients(self):
        pass


class InvalidInputError(Exception):
    pass


def double_sided_t_test_for_coefficients(X: np.matrix, y: np.array, beta: np.array):
    """
    Method to compute the t-statistic for the coefficients of the linear regression model.
    """
    pass


def f_statistic(X: np.matrix, y: np.array, beta: np.array):
    """
    Method to compute the F-statistic for the linear regression model.
    """
    pass

def compute_homoscedasticity(X: np.matrix, y: np.array, beta: np.array):
    """
    Method to compute the homoscedasticity of the linear regression model.
    """
    pass

def check_multicollinearity(X: np.matrix):
    """
    Method to check for multicollinearity in the linear regression model.
    """
    pass

def check_guass_markov_assumptions(X: np.matrix, y: np.array):
    """
    Method to check the Gauss-Markov assumptions for the linear regression model.
    """
    pass

import numpy as np
from numpy.linalg import inv
from utils import check_if_matrix_is_invertible
from models.base import LinearRegressionBase



# first define the model:

class OLSRegression(LinearRegressionBase):
    """Linear Regression Model Class
    This class implements a simple linear regression model.
    This is the OLS closed-form solution (also called the normal equation):
    """

    def __init__(self):

        super().__init__()
        self.coef_ = None         # Coefficients (without intercept)
        self.intercept_ = None    # Intercept
        self.r_squared_ = None

    def __str__(self):
        return f"<LinearRegression: intercept={self.intercept_}, R²={self.r_squared_}>"

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Method to compute the parameters of linear regression and R squared.
        """
        # just to check they are in correct format 
        # X is a matrix of shape (n_samples, n_features)
        # y is a vector of shape (n_samples,)
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array.")
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy array.")

        # need to rewrite:
        # it can only move to this section if the matrix is non-singular. 
        # This formula gives the best linear unbiased estimator (BLUE) of the regression coefficients under the Gauss-Markov assumptions.
        # TODO add an intercept if not present

        check_if_matrix_is_invertible(X.T @ X)


        beta = inv(X.T @ X) @ X.T @ y        
        
        self.coef_  = beta

        print(X.shape)
        print(beta.shape)

        y_hat = X @ beta

        residuals = y - y_hat
        print(f'Residuals shape: { y - y_hat}')
    
        print(y_hat.shape)
        print(residuals.shape)
        squared_residuals = residuals.T @ residuals

        print(f"y.shape: {y.shape}, type: {type(y)}")
        print(f"y_hat.shape: {y_hat.shape}, type: {type(y_hat)}")

        # r squared calculation:
        sum_of_squared_residuals = np.sum(squared_residuals)
        total_sum_of_squares = np.sum((y - np.mean(y)) ** 2)    
        self.r_squared_ = 1 - (sum_of_squared_residuals / total_sum_of_squares)

        return self
    

    def predict(self, X: np.matrix):
        pass


    def r_squared(self, X: np.matrix, y: np.array):
        pass

    def p_values_for_coefficients(self):
        pass




class InvalidInputError(Exception):
    pass


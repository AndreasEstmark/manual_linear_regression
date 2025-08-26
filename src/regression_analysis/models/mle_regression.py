import numpy as np
from numpy.linalg import inv
from ..utils.diagnostics import check_if_matrix_is_invertible
from regression_analysis.models.linear_base import LinearModel



# first define the model:

class MLERegression(LinearModel):
    """Linear Regression Model Class
    This class implements a simple linear regression model."""

    def __init__(self):
        super().__init__()
        self.coef_ = None
        self.intercept_ = None

    def __str__(self):
        return f"LinearRegression: mle"

    def simple_fit (self, X: np.matrix, y: np.array):
        """
        Method to compute the parameters of linear regression and R squared.
        """
        # just to check they are in correct format 
        # X is a matrix of shape (n_samples, n_features)
        # y is a vector of shape (n_samples,)
        if not isinstance(X, np.matrix):
            raise TypeError("X must be a numpy matrix.")
        
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy array.")
        

        # need to rewrite:
        # it can only move to this section if the matrix is non-singular. 
     
        pass

    def predict(self, X: np.matrix):
        pass

    def p_values_for_coefficients(self):
        pass




class InvalidInputError(Exception):
    pass


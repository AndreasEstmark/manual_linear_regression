import numpy as np
from numpy.linalg import inv
from regression_analysis.utils.diagnostics import check_if_matrix_is_invertible
from regression_analysis.models.linear_base import LinearModel
from regression_analysis.utils.diagnostics import *
from regression_analysis.utils.diagnostics_handling import handle_high_vif_columns
from regression_analysis.utils.exceptions import FitError


class OLSRegression(LinearModel):
    """Linear Regression Model Class
    This class implements a simple linear regression model.
    This is the OLS closed-form solution (also called the normal equation):
    """

    def __init__(self):

        super().__init__()
        self.coef_ = None         # Coefficients (without intercept)
        self.intercept = None    # Intercept
        self.r_squared = None

    def __str__(self):
        return f"LinearRegression: ols"

    def simple_fit(self, X: np.ndarray, y: np.ndarray):
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

        # it can only move to this section if the matrix is non-singular. 
        # This formula gives the best linear unbiased estimator (BLUE) of the regression coefficients under the Gauss-Markov assumptions.
        # TODO add an intercept if not present

        check_if_matrix_is_invertible(X.T @ X)

        beta = inv(X.T @ X) @ X.T @ y        
        
        self.coef_  = beta

        y_hat = X @ beta

        residuals = y - y_hat
      
        squared_residuals = residuals.T @ residuals
        # r squared calculation:
        sum_of_squared_residuals = np.sum(squared_residuals) # SSR
        total_sum_of_squares = np.sum((y - np.mean(y)) ** 2)    # TSS 
        self.r_squared = 1 - (sum_of_squared_residuals / total_sum_of_squares) #R_squared

        return self

    def fit_and_diagnostics(self, X: np.ndarray, y: np.ndarray):
        #model = self.simple_fit(X, y)

        vifs = check_multicollinearity_in_regressors(X)

        X = handle_high_vif_columns(X, vifs)

        model = self.simple_fit(X, y)

        compute_double_sided_t_test_for_coefficients(X, y, model.coef_)

        return 

    def fit_and_predict(self, X, y):
        pass

    def predict(self, X: np.ndarray):
        pass

    def p_values_for_coefficients(self):
        pass


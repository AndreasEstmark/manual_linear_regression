from numpy.linalg import inv
import numpy as np
from regression_analysis.utils.diagnostics import check_if_matrix_is_invertible
from abc import ABC, abstractmethod

class LinearRegressionBase(ABC):
    """Abstract base class for linear regression models."""

    def __init__(self):
        super().__init__() # here in the base class this is not strictly needed
        self.coef_ = None
        self.intercept_ = None

    @abstractmethod
    def __str__(self):
        pass
    
    @abstractmethod
    def simple_fit(self, X: np.ndarray, y: np.ndarray):
        pass
    
    def fit_and_diagnostics(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model and run diagnostics.
        """
        pass
    
    def fit_and_predict(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model and make predictions.
        """
        self.fit(X, y)
        return self.predict(X)

    @abstractmethod
    def predict(self, X: np.ndarray,):
        pass

    @abstractmethod
    def r_squared(self):
        pass

    @abstractmethod
    def p_values_for_coefficients(self):
        pass
 
class InvalidInputError(Exception):
    pass


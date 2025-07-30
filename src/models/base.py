import pandas as pd
import numpy as np
from numpy.linalg import inv
from utils import check_if_matrix_is_invertible
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
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def r_squared(self, X, y):
        pass

    @abstractmethod
    def p_values_for_coefficients(self):
        pass



class InvalidInputError(Exception):
    pass


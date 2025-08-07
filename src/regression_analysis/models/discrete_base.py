import numpy as np
from abc import ABC, abstractmethod


class LogisticRegressionBase(ABC):
    """Abstract base class for logistic regression models."""

    def __init__(self):
        super().__init__() # here in the base class this is not strictly needed
        self.coef_ = None
        self.intercept_ = None

    @abstractmethod
    def sigmoid(self, z: np.ndarray):
        """
        Compute the sigmoid function.

        """
        pass
    
    @abstractmethod
    def _logits(self, X: np.ndarray):
        """
        Compute the logits for the input features.

        """
        pass 
    
    @abstractmethod
    def simple_fit(self, X: np.ndarray, y: np.ndarray):
        pass
    
    @abstractmethod
    def fit_and_diagnostics(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model and run diagnostics.
        """
        pass
    
    @abstractmethod
    def fit_and_predict(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model and make predictions.
        """
        self.fit(X, y)
        return self.predict(X)

    @abstractmethod
    def predict(self, X: np.ndarray,):
        pass

 

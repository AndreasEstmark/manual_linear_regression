import numpy as np
from abc import ABC, abstractmethod


from models.base import LogisticRegressionBase  

class LogisticRegression(LogisticRegressionBase):

    def __init__(self):
        super().__init__()


    def __str__(self):
        return f"LogisticRegression:"
    

class LogisticRegression(ABC):

    def __init__(self):
        super().__init__() 
        self.coef_ = None
        self.intercept_ = None

    def __str__(self):
        pass

    def sigmoid(self, z: np.ndarray):
        """
        Compute the sigmoid function.

        """
        return 1 / (1 + np.exp(-z))
    
    def _logits(self, X: np.ndarray):
        """
        Compute the logits for the input features.

        """
        pass 
    
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

    def predict(self, X: np.ndarray,):
        pass

 

import numpy as np
import pytest
from src.regression_analysis.utils import check_if_matrix_is_invertible

def test_invertible_matrix():
    X = np.array([[1, 2], [3, 4.1]])  # Not singular
    assert check_if_matrix_is_invertible(X) == True

def test_singular_matrix_raises_value_error():
    X = np.array([[1, 2], [2, 4]])  # Singular (second row = 2 * first)
    with pytest.raises(ValueError, match="not invertible"):
        check_if_matrix_is_invertible(X)
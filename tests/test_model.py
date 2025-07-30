import numpy as np
import pytest
from model import LinearRegression
from utils import check_if_matrix_is_invertible


# old code
# Test LinearRegression fit method with a simple dataset
def test_fit_returns_correct_shape():
    X = np.matrix([[1, 2], [1, 3], [1, 4]])
    y = np.array([2, 3, 4])
    model = LinearRegression()
    beta = model.fit(X, y)
    assert beta.shape == (2,)

# Test that fit raises TypeError for wrong X type
def test_fit_raises_typeerror_for_wrong_X():
    X = [[1, 2], [1, 3], [1, 4]]  # Not a np.matrix
    y = np.array([2, 3, 4])
    model = LinearRegression()
    with pytest.raises(TypeError):
        model.fit(X, y)

# Test check_if_matrix_is_invertible raises ValueError for singular matrix
def test_check_if_matrix_is_invertible_raises():
    X = np.matrix([[1, 2], [2, 4]])  # Singular matrix
    with pytest.raises(ValueError):
        check_if_matrix_is_invertible(X)

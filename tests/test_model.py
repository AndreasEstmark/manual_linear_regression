import numpy as np
import pytest
from regression_analysis.models.ols import OLSRegression


# old code
# Test ols fit method with a simple dataset
def test_ols_fit_returns_correct_shape():
    X = np.array([[1, 2], [1, 3], [1, 4]])
    y = np.array([2, 3, 4])
    model = OLSRegression()
    model.simple_fit(X, y)
    assert model.coef_.shape == (2,)

# Test ols that fit raises TypeError for wrong X type
def test_ols_fit_raises_typeerror_for_wrong_X():
    X = [[1, 2], [1, 3], [1, 4]]  # Not a np.array
    y = np.array([2, 3, 4])
    model = OLSRegression()
    with pytest.raises(TypeError):
        model.simple_fit(X, y)

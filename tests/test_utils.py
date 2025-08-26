import numpy as np
import pytest
from regression_analysis.utils.diagnostics import check_if_matrix_is_invertible

from unittest.mock import patch, MagicMock
from numpy.linalg import LinAlgError

from regression_analysis.utils.diagnostics import check_multicollinearity_in_regressors
from regression_analysis.utils.exceptions import MulticollinearityError

def test_invertible_matrix():
    X = np.array([[1, 2], [3, 4.1]])  # Not singular
    assert check_if_matrix_is_invertible(X) == True

def test_singular_matrix_raises_value_error():
    X = np.array([[1, 2], [2, 4]])  # Singular (second row = 2 * first)
    with pytest.raises(ValueError, match="not invertible"):
        check_if_matrix_is_invertible(X)


def test_check_multicollinearity_wraps_linalgerror():
    X = np.ones((5, 2))  # dummy matrix, actual values don't matter

    # Patch OLSRegression in the models.ols module where it's defined so imports return our mock
    with patch("regression_analysis.models.ols.OLSRegression") as MockOLS:
        # Mock .simple_fit to raise LinAlgError
        instance = MockOLS.return_value
        instance.simple_fit.side_effect = LinAlgError("Singular matrix")

        # Now our function should re-raise as MulticollinearityError
        with pytest.raises(MulticollinearityError) as excinfo:
            check_multicollinearity_in_regressors(X)

        # Check that our exception contains context from LinAlgError
        assert "Singular matrix" in str(excinfo.value)
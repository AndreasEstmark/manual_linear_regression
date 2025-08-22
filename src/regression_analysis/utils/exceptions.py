
class MulticollinearityError(Exception):
    """Raised when multicollinearity is detected and can't be fixed."""
    pass

class VIFHandlingError(Exception):
    """Raised when VIF-based column handling fails."""
    pass

class ModelFitError(Exception):
    """Raised when model fitting fails."""
    pass

class DiagnosticsError(Exception):
    """Raised when statistical diagnostics fail."""
    pass    
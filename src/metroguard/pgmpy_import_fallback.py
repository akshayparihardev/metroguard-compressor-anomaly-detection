"""Let pgmpy import when Windows Application Control blocks statsmodels' time-series DLLs."""

import sys
import types


def install_statsmodels_api_fallback() -> bool:
    """Register a minimal `statsmodels.api` only if the real one cannot be imported.

    pgmpy imports `statsmodels.api` for its linear-Gaussian models, which this project never uses.
    On machines where Smart App Control blocks `statsmodels.tsa` DLLs the import fails and takes
    pgmpy down with it. The fallback exposes the three regression classes pgmpy asks for, taken
    from `statsmodels.regression.linear_model`, which loads fine. Returns True if it was needed.
    """
    try:
        import statsmodels.api  # noqa: F401
    except ImportError:
        from statsmodels.regression.linear_model import GLS, OLS, WLS

        fallback = types.ModuleType("statsmodels.api")
        fallback.GLS, fallback.OLS, fallback.WLS = GLS, OLS, WLS
        sys.modules["statsmodels.api"] = fallback
        return True
    return False

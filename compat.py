"""
Compatibility helpers for loading sklearn models pickled with older versions.

The model.pkl was saved with scikit-learn 1.1.3 which uses a tree node dtype
WITHOUT the 'missing_go_to_left' field (added in sklearn 1.4+). This module
monkey-patches the internal `_check_node_ndarray` function so it upgrades
old-format node arrays on-the-fly.
"""

import warnings
import numpy as np

# Suppress InconsistentVersionWarning when loading old pickle files
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", message=".*Trying to unpickle.*")

try:
    import sklearn.tree._tree as _tree_mod

    _original_check = _tree_mod._check_node_ndarray

    def _patched_check(node_ndarray, expected_dtype):
        """
        If the node array is missing the 'missing_go_to_left' field,
        add it with a default value of 1 (True) before validation.
        """
        actual_names = set(node_ndarray.dtype.names or [])
        expected_names = set(expected_dtype.names or [])
        missing = expected_names - actual_names

        if "missing_go_to_left" in missing:
            new_arr = np.zeros(node_ndarray.shape[0], dtype=expected_dtype)
            for name in node_ndarray.dtype.names:
                new_arr[name] = node_ndarray[name]
            new_arr["missing_go_to_left"] = 1
            return _original_check(new_arr, expected_dtype)

        return _original_check(node_ndarray, expected_dtype)

    _tree_mod._check_node_ndarray = _patched_check

except Exception:
    pass  # If patching fails, let the original error propagate

# --- Patch 2: Add missing 'monotonic_cst' attribute ---
# sklearn 1.4+ expects DecisionTreeClassifier to have a monotonic_cst attribute,
# but models pickled with 1.1.3 don't have it.
try:
    from sklearn.tree import DecisionTreeClassifier

    _orig_getattr = DecisionTreeClassifier.__getattribute__

    def _patched_getattr(self, name):
        if name == "monotonic_cst":
            try:
                return _orig_getattr(self, name)
            except AttributeError:
                return None
        return _orig_getattr(self, name)

    DecisionTreeClassifier.__getattribute__ = _patched_getattr
except Exception:
    pass

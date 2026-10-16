"""Scikit-learn-compatible local radial-basis-function regression.

The numerical kernel solver is SciPy's RBFInterpolator. This is deterministic
kernel interpolation, not a Gaussian process with posterior uncertainty.
"""
import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted


class LocalRBFRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, neighbors=48, kernel="quintic", degree=2):
        self.neighbors = neighbors
        self.kernel = kernel
        self.degree = degree

    def fit(self, X, y):
        X = check_array(X)
        y = check_array(y, ensure_2d=True)
        if len(X) != len(y):
            raise ValueError("X and y lengths differ")
        self.n_features_in_ = X.shape[1]
        self.scaler_ = StandardScaler().fit(y)
        self.interpolator_ = RBFInterpolator(
            X, self.scaler_.transform(y), neighbors=self.neighbors,
            kernel=self.kernel, degree=self.degree, smoothing=0)
        return self

    def predict(self, X):
        check_is_fitted(self, "interpolator_")
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Incorrect input feature count")
        return self.scaler_.inverse_transform(self.interpolator_(X))

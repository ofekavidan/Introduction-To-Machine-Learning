from __future__ import annotations
from typing import Callable
from typing import NoReturn
from ...base import BaseEstimator
from ...metrics import misclassification_error
import numpy as np


def default_callback(fit: Perceptron, x: np.ndarray, y: int):
    pass


class Perceptron(BaseEstimator):
    """
    Perceptron half-space classifier

    Finds a separating hyperplane for given linearly separable data.

    Attributes
    ----------
    include_intercept: bool, default = True
        Should fitted model include an intercept or not

    max_iter_: int, default = 1000
        Maximum number of passes over training data

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by Perceptron algorithm. To be set in
        `Perceptron.fit` function.

    callback_: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response
    """

    def __init__(self, include_intercept: bool = True, max_iter: int = 1000,
                 callback: Callable[[Perceptron, np.ndarray, int], None] = default_callback):
        """
        Instantiate a Perceptron classifier

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        max_iter: int, default = 1000
            Maximum number of passes over training data

        callback: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response
        """

        super().__init__()

        self.include_intercept_ = include_intercept
        self.max_iter_ = max_iter
        self.callback_ = callback
        self.coefs_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a halfspace to the given samples. Iterate over given data as long as there exists a sample misclassified
        or that did not reach `self.max_iter_`

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.fit_intercept_`
        """

        self.fitted_ = True

        # If we are required to include an intercept, we'll add another 'feature' which will be a column
        # of ones, and add another coefficient for the intercept itself
        if self.include_intercept_:
            X = np.c_[np.ones(X.shape[0]), X]

        self.coefs_ = np.zeros(shape=(X.shape[1]))

        # For as long as we didn't reach the max iterations, and didn't find a "perfect" coefficients vector
        # we want to update our coefficients to get better results.
        for _ in range(self.max_iter_):

            # If we don't have any errors, we are good to go
            if misclassification_error(self._predict(X), y) == 0:
                return self

            # Otherwise, loop over all samples and if we didn't get the current one right, adjust the coefficients
            for i in range(X.shape[0]):
                if y[i] * (X[i] @ self.coefs_) <= 0:
                    self.coefs_ = self.coefs_ + (y[i] * X[i])
                    self.callback_(self, X[i], y[i])
                    break

        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        if self.include_intercept_ and X.shape[1] != self.coefs_.shape[0]:
            X = np.c_[np.ones(X.shape[0]), X]

        return np.sign(X @ self.coefs_)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """

        return misclassification_error(y, self._predict(X))

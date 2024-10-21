from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        error = None

        # Find the best threshold out of (every feature, with signs 1 and -1)
        for j, sign in product(range(X.shape[1]), [-1, 1]):
            thr, thr_err = self._find_threshold(X[:, j], y, sign)

            if error is None or thr_err < error:
                error = thr_err

                self.threshold_ = thr
                self.j_ = j
                self.sign_ = sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """

        feature_column = X[:, self.j_]

        return np.where(feature_column < self.threshold_, -self.sign_, self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """

        # Sorting the values and labels according to the values in an increasing order
        sorted_indexes = np.argsort(values)
        values, labels = values[sorted_indexes], labels[sorted_indexes]

        # Calculating the possible thresholds following this principle:
        #   - First threshold will be -infinity  =>  all elements will be classified as +sign
        #   - For all i in [1, len(values)-1], the i'th threshold will be the i'th value,
        #     meaning every element lesser than the i'th threshold will be classified as -sign
        #   - Last threshold will be +infinity  =>  all elements will be classified as -sign
        thresholds = np.concatenate([[-np.inf], values[1:], [np.inf]])

        # Calculate the number of values that we correctly classified if we took the threshold to be -infinity,
        # taking the values' weights into account (for later usage in AdaBoost)
        correct_min_threshold = np.sum(np.abs(labels * (np.sign(labels) == sign)))

        # After we got the correctness for threshold being -infinity, we use #np.cumsum to calculate
        # the number of (weighted) elements that were changed in a good/bad way after moving the threshold.
        # This being: From all the (weighted) elements we classified correctly, reduce, for each index in the
        # #np.cumsum result, the number of (weighted) elements that are now classified correctly.
        3
        # If we ended up changing more (weighted) elements to the wrong sign, we will reduce a negative value, meaning
        # this is a bad index to take for a threshold (since we want the minimal).
        # The more (weighted) elements we got correctly, the higher the cumsum number is, the smaller value we'll have
        # => the better the threshold is.
        incorrect_per_threshold = np.append(correct_min_threshold,
                                            correct_min_threshold - np.cumsum(labels * sign))

        # Taking the index of the element with the minimal value would result in taking the threshold that
        # reduced the most (weighted) correct elements to the base threshold (-infinity)
        #   If the index is 0, it means that we classified correctly the most (weighted) elements when
        #   we classified all of them as +sign
        #   If the index is len(values), it means that we classified correctly the most (weighted) elements
        #   when we classified all of them as -sign
        best_threshold_index = np.argmin(incorrect_per_threshold)

        # Returning the threshold value at the best index, and it's error value
        best_threshold = thresholds[best_threshold_index]
        incorrect_values = incorrect_per_threshold[best_threshold_index]

        return best_threshold, incorrect_values

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

        return misclassification_error(y, self.predict(X))

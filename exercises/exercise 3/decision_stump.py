from __future__ import annotations
from typing import Tuple, NoReturn
from base_estimator import BaseEstimator
import numpy as np
from itertools import product
from loss_functions import misclassification_error


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
        Fit a decision stump to the given data. That is, finds the best feature and threshold by which to split

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # Initialize the best loss to a very large number (infinity)
        best_loss = float('inf')

        # Initialize the best feature index, threshold, and sign to None
        best_feature_index = None
        best_threshold_value = None
        best_sign_value = None

        # Get the number of features from the input data
        n_samples, n_features = X.shape

        # Iterate over each feature to find the one that best separates the classes
        for feature_index in range(n_features):
            # Extract the values of the current feature from the dataset
            feature_values = X[:, feature_index]

            # Check both possible signs for the threshold (1 and -1)
            for sign_value in [1, -1]:
                # Find the best threshold and corresponding loss for the current feature and sign
                threshold_value, loss = self._find_threshold(feature_values, y, sign_value)

                # If the current threshold and sign combination results in a lower loss, update the best parameters
                if loss < best_loss:
                    best_loss = loss
                    best_feature_index = feature_index
                    best_threshold_value = threshold_value
                    best_sign_value = sign_value

                # If a perfect threshold is found, stop early from the inner loop
                if best_loss == 0:
                    break

            # If a perfect threshold is found, stop early from the outer loop
            if best_loss == 0:
                break

        # Set the best found feature index, threshold, and sign to the class attributes
        self.j_ = best_feature_index
        self.threshold_ = best_threshold_value
        self.sign_ = best_sign_value

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sign responses for given samples using fitted estimator

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
        # Extract the feature column from the input data that corresponds to the best feature found during fitting
        selected_feature_column = X[:, self.j_]

        # Predict the responses based on whether the selected feature values are below or above the threshold
        predicted_responses = np.where(selected_feature_column < self.threshold_, -self.sign_, self.sign_)

        # Return the predicted responses
        return predicted_responses
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
        # Sort the feature values and corresponding labels
        sorted_indices = np.argsort(values)
        sorted_feature_values = values[sorted_indices]
        sorted_labels = labels[sorted_indices]

        # Calculate the initial loss for the threshold being -infinity
        initial_loss = np.sum(np.abs(sorted_labels)[np.sign(sorted_labels) == sign])

        # Calculate the cumulative loss for each threshold
        cumulative_loss = np.append(initial_loss, initial_loss - np.cumsum(sorted_labels * sign))

        # Find the index of the threshold with the minimum cumulative loss
        best_threshold_index = np.argmin(cumulative_loss)
        min_misclassification_error = float(cumulative_loss[best_threshold_index])  # Ensure the error is a float

        # Determine the best threshold value
        if best_threshold_index == len(sorted_feature_values):
            best_threshold = float('inf')
        elif best_threshold_index == 0:
            best_threshold = float('-inf')
        else:
            best_threshold = float(sorted_feature_values[best_threshold_index])

        # Return the best threshold and its corresponding misclassification error
        return best_threshold, min_misclassification_error

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
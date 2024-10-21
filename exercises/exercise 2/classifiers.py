from __future__ import annotations
from typing import Callable
from typing import NoReturn
from base_estimator import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


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
    def __init__(self,
                 include_intercept: bool = True,
                 max_iter: int = 1000,
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
        Fit a halfspace to to given samples. Iterate over given data as long as there exists a sample misclassified
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
        # Mark the model as fitted
        self.fitted_ = True

        # Include intercept if necessary
        if self.include_intercept_:
            X = np.c_[np.ones(len(X)), X]

        # Initialize coefficients to zeros
        self.coefs_ = np.zeros(X.shape[1])

        # Iterate over the data up to a maximum number of iterations
        for iteration in range(self.max_iter_):
            misclassified_sample_found = False

            # Check each sample
            for idx in range(y.shape[0]):
                product = y[idx] * (self.coefs_ @ X[idx])

                # If a sample is misclassified, update the coefficients
                if product <= 0:
                    self.coefs_ += y[idx] * X[idx]
                    self.callback_(self, X[idx], y[idx])
                    misclassified_sample_found = True
                    break

            # If no misclassified sample was found, stop the iterations
            if not misclassified_sample_found:
                break
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
        X = np.hstack([np.ones([X.shape[0], 1]), X]) if self.include_intercept_ else X
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
        from loss_functions import misclassification_error
        return misclassification_error(y, y_pred=self._predict(X))



class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        # Get unique classes and their counts
        self.classes_, class_counts = np.unique(y, return_counts=True)

        # Estimate class priors
        self.pi_ = class_counts / len(y)

        # Initialize the covariance matrix
        self.cov_ = np.zeros((X.shape[1], X.shape[1]))

        # Initialize list to store means for each class
        means = []

        # Iterate over each class
        for cls in self.classes_:
            # Compute mean vector for the current class
            cls_mean = X[y == cls].mean(axis=0)
            means.append(cls_mean)

            # Compute deviations from the mean for the current class
            deviations = X[y == cls] - cls_mean

            # Compute class covariance contribution and add to the total covariance matrix
            self.cov_ += np.dot(deviations.T, deviations)

        # Stack the means vertically to form the mu_ matrix
        self.mu_ = np.vstack(means)

        # Normalize the covariance matrix
        self.cov_ /= (len(y) - len(self.classes_))

        # Compute the inverse of the covariance matrix
        self._cov_inv = np.linalg.inv(self.cov_)



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
        return self.classes_[np.argmax(self.likelihood(X),axis=1)]



    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # Initialize the result array to store likelihoods
        likelihoods = np.zeros((n_samples, n_classes))

        # Iterate over each class to compute the likelihood
        for idx, cls in enumerate(self.classes_):
            # Compute the difference between X and the class mean
            mean_diff = X - self.mu_[idx]

            # Compute the Mahalanobis distance
            mahalanobis_dist = np.sum(mean_diff @ self._cov_inv * mean_diff, axis=1)

            # Compute the Gaussian likelihood
            coeff = 1 / np.sqrt((2 * np.pi) ** n_features * np.linalg.det(self.cov_))
            exponent = np.exp(-0.5 * mahalanobis_dist)

            # Multiply by class prior probability
            likelihoods[:, idx] = coeff * exponent * self.pi_[idx]

        return likelihoods




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
        from loss_functions import misclassification_error
        return misclassification_error(y,self.predict(X))

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # Determine unique classes and initialize parameters
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # Initialize mean, variance, and class prior arrays
        self.mu_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))
        self.pi_ = np.zeros(n_classes)

        # Calculate mean, variance, and class prior for each class
        for idx, cls in enumerate(self.classes_):
            # Select samples belonging to the current class
            X_class = X[y == cls]

            # Calculate mean and variance of the current class
            self.mu_[idx, :] = np.mean(X_class, axis=0)
            self.vars_[idx, :] = np.var(X_class, axis=0, ddof=1)  # Use unbiased estimator for variance

            # Calculate prior probability of the current class
            self.pi_[idx] = X_class.shape[0] / X.shape[0]

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
        likelihoods = self.likelihood(X)
        return self.classes_[np.argmax(likelihoods, axis=1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        likelihoods = np.zeros((n_samples, n_classes))

        # Calculate the likelihood for each class
        for idx, cls in enumerate(self.classes_):
            mean = self.mu_[idx]
            var = self.vars_[idx]
            prior = self.pi_[idx]

            # Calculate Gaussian likelihood component-wise
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var))
            log_likelihood -= 0.5 * np.sum(((X - mean) ** 2) / var, axis=1)

            # Combine with the log prior probability
            likelihoods[:, idx] = log_likelihood + np.log(prior)

        return likelihoods

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
        from loss_functions import misclassification_error
        y_pred = self._predict(X)
        return misclassification_error(y, y_pred)

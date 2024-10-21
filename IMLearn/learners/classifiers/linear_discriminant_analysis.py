from typing import NoReturn
from ...base import BaseEstimator
from ...metrics import misclassification_error
import numpy as np
from numpy.linalg import det, inv


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

        # Saves unique y values and the number of appearances each have (normalized)
        self.classes_, self.pi_ = np.unique(y, return_counts=True)
        self.pi_ = self.pi_ / len(y)

        # According to the MLE mu defined in Practice 6.
        # Basically, summing up all Xi s.t. yi is in the current class, and then taking its mean
        # Results in a vector for each class, therefore a matrix of mu
        self.mu_ = np.array([np.mean(X[y == clazz], axis=0) for clazz in self.classes_])

        # According to the MLE cov defined in Practice 6.
        self.cov_ = np.sum([(X[y == clazz] - self.mu_[i]).T @ (X[y == clazz] - self.mu_[i])
                            for clazz, i in enumerate(self.classes_)], axis=0) / (len(X) - len(self.classes_))

        self._cov_inv = inv(self.cov_)

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

        return self.classes_[np.argmax(self.likelihood(X), axis=1)]

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

        likelihoods = np.zeros(shape=(len(X), len(self.classes_)))

        # For each sample, calculate its pdf with each class' parameters (Mu & Cov)
        for i in range(len(X)):
            for k in range(len(self.classes_)):
                likelihoods[i, k] = np.power(2 * np.pi, -len(X[i]) * 0.5) * np.power(det(self.cov_), -0.5) * \
                        np.exp(-0.5 * (X[i] - self.mu_[k]).T @ self._cov_inv @ (X[i] - self.mu_[k])) * \
                        self.pi_[k]

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

        return misclassification_error(y, self.predict(X))

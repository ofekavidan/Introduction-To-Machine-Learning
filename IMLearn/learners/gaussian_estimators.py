from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False):
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """

        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """

        self.mu_ = X.mean()

        # If self.biased_, we want to use a biased variance estimator.
        # In class (Course book, page 13) we saw that a biased variance estimator is: (1 / m) * sum((Xi - u)^2)
        #                                      and an unbiased variance estimator is: (1 / (m - 1)) * sum((Xi - u)^2)

        # We can use 'ddof' (which affects the denominator) to achieve biased/unbiased variance.
        # * The denominator used by numpy.ndarray.var is: "m - ddof"
        self.var_ = X.var(ddof=0) if self.biased_ else X.var(ddof=1)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """

        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        # The PDF of a normal distribution is (Course book, page 11):
        # (1 / sqrt(2 * pi * sigma^2)) * exp((-1 / 2 * sigma^2) * (X - mu)^2)

        # * Notice that X is a random vector. Thus, when using np.power(X - self.mu_, 2), we square each
        #   random variable (each entry) individually. As a result, we end up with an m-long vector

        coefficient = 1 / np.sqrt(2 * np.pi * self.var_)
        exp_coefficient = -1 / (2 * self.var_)
        exp_error_squared = np.power(X - self.mu_, 2)

        return coefficient * np.exp(exp_coefficient * exp_error_squared)

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """

        # To calculate log-likelihood, we can calculate the likelihood and then extract its logarithm.
        # The normal distribution likelihood function is (Course book, page 15):
        # (1 / (2 * pi * sigma^2)^(m/2)) * exp((-1/2 * sigma) * sum(xi - mu)^2)

        coefficient = 1 / np.power(2 * np.pi * sigma, len(X) / 2)
        exp_coefficient = -1 / (2 * sigma)
        exp_error_sum = np.sum(np.power(X - mu, 2))

        return np.log(coefficient * np.exp(exp_coefficient * exp_error_sum))


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """

        self.fitted_, self.mu_, self.cov_ = False, None, None

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """

        # The mean in a Multivariate Gaussian distribution, is a vector (of length D) of the means.
        # We can achieve this by calculating the mean over the 0 axis.
        self.mu_ = X.mean(axis=0)

        # The covariance matrix is a DxD matrix defined as follows:
        # cov_[i, j] = (1 / (m - 1)) * sum((X[k][i] - u[i]) * (x[k][j] - u[j])
        # Notice that we can also write cov_ as: (1 / (m - 1)) * (X - u)^T * (X - u)
        # Using np.cov, we can get this result easily. It computes the average (mean) of X in each dimension,
        # then calculates (X - u) and computes its dot product, dividing by (m - 1)
        self.cov_ = np.cov(X, rowvar=False, bias=False)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """

        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        # The PDF of a normal, multivariate distribution is (Course book, page 23):
        # (1 / sqrt((2 * pi)^d * |cov|)) * exp((-1 / 2) * (X - mu)^t * (cov)^-1 * (X - mu))
        # * Notice that X is a random vector, each entry is a vector of size D.

        d = len(self.mu_)
        coefficient = 1 / (np.sqrt(np.power(2 * np.pi, d)) * det(self.cov_))
        exp_coefficient = (-1 / 2) * (np.sum((X - self.mu_) @ inv(self.cov_) * (X - self.mu_), axis=1))

        return coefficient * np.exp(exp_coefficient)

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """

        # To calculate log-likelihood, we can use the formula from Q13 in this exercise (similar to wikipedia's)
        # The normal, multivariate distribution log-likelihood function is:
        # (-m / 2) * (ln(|cov|) + ln((2 * pi)^d)) + (-1 / 2) * sum((Xi - u)^T * cov^-1 * (Xi - u)))

        m = len(X)
        d = len(mu)

        first_term = (-m / 2) * (slogdet(cov)[1] + d * np.log(2 * np.pi))
        # (X - mu) is a Matrix of size 1000x4. Therefore, it's multiplication with inv(cov) (4x4 matrix) is
        # well-defined. We then want to multiply it by terms with (X - mu), resulting in the desired sum.
        second_term = (-1 / 2) * np.sum((X - mu) @ inv(cov) * (X - mu))

        return first_term + second_term

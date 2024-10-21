from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """

    shuffled = np.arange(len(X))
    np.random.shuffle(shuffled)

    sets = np.array_split(shuffled, cv)

    train_score = validation_score = 0

    for set_ids in sets:
        remaining_ids = np.delete(np.arange(len(X)), set_ids)

        train_X, train_y = X[remaining_ids], y[remaining_ids]

        estimator.fit(train_X, train_y)

        train_score += scoring(train_y, estimator.predict(train_X))
        validation_score += scoring(y[set_ids], estimator.predict(X[set_ids]))

    return (train_score/cv), (validation_score/cv)

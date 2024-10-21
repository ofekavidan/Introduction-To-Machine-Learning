from __future__ import annotations
from typing import Callable, NoReturn
import numpy as np

from IMLearn.base import BaseModule, BaseLR
from .learning_rate import FixedLR

OUTPUT_VECTOR_TYPE = ["last", "best", "average"]


def default_callback(**kwargs) -> NoReturn:
    pass


class GradientDescent:
    """
    Gradient Descent algorithm

    Attributes:
    -----------
    learning_rate_: BaseLR
        Learning rate strategy for retrieving the learning rate at each iteration t of the algorithm

    tol_: float
        The stopping criterion. Training stops when the Euclidean norm of w^(t)-w^(t-1) is less than
        specified tolerance

    max_iter_: int
        The maximum number of GD iterations to be performed before stopping training

    out_type_: str
        Type of returned solution:
            - `last`: returns the point reached at the last GD iteration
            - `best`: returns the point achieving the lowest objective
            - `average`: returns the average point over the GD iterations

    callback_: Callable[[...], None], default=default_callback
        A callable function to be called after each update of the model while fitting to given data.
        Callable function receives as input any argument relevant for the current GD iteration. Arguments
        are specified in the `GradientDescent.fit` function
    """

    def __init__(self,
                 learning_rate: BaseLR = FixedLR(1e-3),
                 tol: float = 1e-5,
                 max_iter: int = 1000,
                 out_type: str = "last",
                 callback: Callable[[GradientDescent, ...], None] = default_callback):
        """
        Instantiate a new instance of the GradientDescent class

        Parameters
        ----------
        learning_rate: BaseLR, default=FixedLR(1e-3)
            Learning rate strategy for retrieving the learning rate at each iteration t of the algorithm

        tol: float, default=1e-5
            The stopping criterion. Training stops when the Euclidean norm of w^(t)-w^(t-1) is less than
            specified tolerance

        max_iter: int, default=1000
            The maximum number of GD iterations to be performed before stopping training

        out_type: str, default="last"
            Type of returned solution. Supported types are specified in class attributes

        callback: Callable[[...], None], default=default_callback
            A callable function to be called after each update of the model while fitting to given data.
            Callable function receives as input any argument relevant for the current GD iteration. Arguments
            are specified in the `GradientDescent.fit` function
        """

        self.learning_rate_ = learning_rate

        if out_type not in OUTPUT_VECTOR_TYPE:
            raise ValueError("output_type not supported")

        self.out_type_ = out_type
        self.tol_ = tol
        self.max_iter_ = max_iter
        self.callback_ = callback

    def fit(self, f: BaseModule, X: np.ndarray, y: np.ndarray):
        """
        Optimize module using Gradient Descent iterations over given input samples and responses

        Parameters
        ----------
        f : BaseModule
            Module of objective to optimize using GD iterations
        X : ndarray of shape (n_samples, n_features)
            Input data to optimize module over
        y : ndarray of shape (n_samples, )
            Responses of input data to optimize module over

        Returns
        -------
        solution: ndarray of shape (n_features)
            Obtained solution for module optimization, according to the specified self.out_type_

        Notes
        -----
        - Optimization is performed as long as self.max_iter_ has not been reached and that
        Euclidean norm of w^(t)-w^(t-1) is more than the specified self.tol_

        - At each iteration the learning rate is specified according to self.learning_rate_.lr_step

        - At the end of each iteration the self.callback_ function is called passing self and the
        following named arguments:
            - solver: GradientDescent
                self, the current instance of GradientDescent
            - weights: ndarray of shape specified by module's weights
                Current weights of objective
            - val: ndarray of shape specified by module's compute_output function
                Value of objective function at current point, over given data X, y
            - grad:  ndarray of shape specified by module's compute_jacobian function
                Module's jacobian with respect to the weights and at current point, over given data X,y
            - t: int
                Current GD iteration
            - eta: float
                Learning rate used at current iteration
            - delta: float
                Euclidean norm of w^(t)-w^(t-1)

        """

        # Initiating first weights vector
        best_weight, best_val = f.weights, f.compute_output(X=X, y=y)
        last_weight = f.weights
        mean_weight = f.weights
        iterations = 0

        weight_t = f.weights

        for t in range(self.max_iter_):
            # Calculate the gradient and eta to be used to update current weight
            grad = f.compute_jacobian(X=X, y=y)
            eta = self.learning_rate_.lr_step(t=t)

            # Update current weights vector
            f.weights = weight_t - eta * grad

            # Calculate the performance and delta of the new weights vector
            val = f.compute_output(X=X, y=y)
            delta = np.linalg.norm(f.weights - weight_t, ord=2)

            # Call the callback function
            self.callback_(solver=self, weights=f.weights, val=val, grad=grad, t=t, eta=eta, delta=delta)

            # Update the last weights vector, mean weights vector and best weights vector
            last_weight = f.weights
            mean_weight = f.weights if mean_weight is None else (mean_weight + f.weights)
            best_weight = f.weights if best_weight is None else (f.weights if best_val < val else best_weight)
            best_val = val if best_val is None else (val if best_val < val else best_val)

            iterations = iterations + 1

            # Update the weight_t to be the weights vector calculated at this iteration
            weight_t = f.weights

            # If the delta (change between weight vectors) is less than tol, break the loop
            if delta < self.tol_:
                break

        # Return the correct weights vector according to out type
        if self.out_type_ == "last":
            return last_weight
        elif self.out_type_ == "best":
            return best_weight
        else:
            return mean_weight / iterations
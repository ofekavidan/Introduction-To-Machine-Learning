import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from sklearn import metrics
from base_module import BaseModule
from base_learning_rate import BaseLR
from gradient_descent import GradientDescent
from learning_rate import FixedLR
from sklearn.metrics import roc_curve, roc_auc_score, auc
from modules import L1, L2
from logistic_regression import LogisticRegression
from utils import split_train_test
from cross_validate import cross_validate
from loss_functions import misclassification_error

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    # Initialize list to store recorded objective values
    objective_values = []

    # Initialize list to store recorded weight parameters
    weight_parameters = []

    # Define the callback function to be used during gradient descent
    def gd_callback(solver, weights, val, grad, t, eta, delta):
        # Append the current objective value to the list
        objective_values.append(val)

        # Append the current weight parameters to the list
        weight_parameters.append(weights)

    # Return the callback function and the lists for recorded values and weights
    return gd_callback, objective_values, weight_parameters


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    # Ensure the output directory exists
    output_dir = './ex4_graphs'
    os.makedirs(output_dir, exist_ok=True)

    # Create an empty array for inputs as we are not using actual data
    empty = np.empty(shape=(0,))

    # Iterate over each learning rate
    for eta in etas:
        # Iterate over each module type (L1 and L2)
        for module, name in [(L1, "L1"), (L2, "L2")]:
            # Get the callback function and lists for recording values and weights
            gd_callback, values, weights = get_gd_state_recorder_callback()

            # Initialize gradient descent with the current learning rate
            gd = GradientDescent(learning_rate=FixedLR(eta), callback=gd_callback)

            # Fit the model using gradient descent
            gd.fit(module(weights=init), X=empty, y=empty)

            # Plot the descent path for the current module and learning rate
            fig = plot_descent_path(module=module,
                                    descent_path=np.array(weights),
                                    title=f"of an {name} module with fixed learning rate η={eta}")

            # Save the plot to a file
            fig.write_html(f"./ex4_graphs/{name}_{eta}_descent.html")

            # Plot the convergence rate for the current module and learning rate
            fig = go.Figure([go.Scatter(y=values, mode="markers")],
                            layout=go.Layout(title=f"Convergence Rate of an {name} module with "
                                                   f"fixed learning rate η={eta}"))

            # Update layout of the plot
            fig.update_layout(width=650, height=500) \
                .update_xaxes(title_text="Iteration") \
                .update_yaxes(title_text=f"Convergence (w {name} norm)")

            # Save the plot to a file
            fig.write_html(f"./ex4_graphs/{name}_{eta}_convergence.html")

            # Print the lowest loss achieved during training
            print(f"The lowest loss achieved by an {name} module with a fixed learning rate η={eta} "
                  f"is {np.round(np.min(values), 13)}")


def load_data(path: str = "SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Ensure the output directory exists
    output_dir = './ex4_graphs'
    os.makedirs(output_dir, exist_ok=True)

    # Load and split SA Heart Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Initialize logistic regression model
    callback,_,_=get_gd_state_recorder_callback()
    model = LogisticRegression(solver=GradientDescent(callback=callback))

    # Fit the model on the training data
    model.fit(X_train.to_numpy(), y_train.to_numpy())

    # Predict probabilities for the training set
    probabilities_train = model.predict_proba(X_train.to_numpy())

    # Plot ROC curve for training data
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, probabilities_train)
    roc_auc_train = auc(fpr_train, tpr_train)

    fig_train = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'), showlegend=False),
                           go.Scatter(x=fpr_train, y=tpr_train, mode='lines', name=f'ROC Curve (AUC = {roc_auc_train:.2f})')])
    fig_train.update_layout(title=f'ROC Curve (AUC = {roc_auc_train:.2f})', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    fig_train.write_html(os.path.join(output_dir, 'gd_logistic_roc_lr0.0001_train.html'))

    # Predict probabilities for the test set
    probabilities_test = model.predict_proba(X_test.to_numpy())

    # Plot ROC curve for test data
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, probabilities_test)
    roc_auc_test = auc(fpr_test, tpr_test)

    fig_test = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'), showlegend=False),
                          go.Scatter(x=fpr_test, y=tpr_test, mode='lines', name=f'ROC Curve (AUC = {roc_auc_test:.2f})')])
    fig_test.update_layout(title=f'ROC Curve (AUC = {roc_auc_test:.2f})', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    fig_test.write_html(os.path.join(output_dir, 'gd_logistic_roc_lr0.0001_test.html'))

    # Find optimal alpha
    optimal_alpha_index = np.argmax(tpr_test - fpr_test)
    optimal_alpha = thresholds_test[optimal_alpha_index]
    print(f"Optimal alpha: {optimal_alpha}")

    # Test error with optimal alpha
    model.alpha_ = optimal_alpha
    test_error = model.loss(X_test.to_numpy(), y_test.to_numpy())
    print(f"Test error with optimal alpha: {test_error}")

    # Fit l1-regularized logistic regression with cross-validation to choose lambda
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    scores= np.zeros((len(lambdas),2))
    for i,lam in enumerate(lambdas):
        model = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(0.0001), max_iter=20000), penalty="l1", lam=lam, alpha=0.5)
        scores[i]=cross_validate(model,X_train.to_numpy(),y_train.to_numpy(),misclassification_error)

    best_lambda= lambdas[np.argmin(scores[:,1])]
    print(f"Best lambda: {best_lambda}")

    # Fit final model with the best lambda
    model = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(0.0001), max_iter=20000), penalty="l1", lam=best_lambda, alpha=0.5)
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    test_error = model.loss(X_test.to_numpy(), y_test.to_numpy())
    print(f"Test error with best lambda: {test_error}")

    fig = go.Figure([go.Scatter(x=lambdas, y=scores[:, 0], name="Train Error"),
                     go.Scatter(x=lambdas, y=scores[:, 1], name="Validation Error")],
                    layout=go.Layout(title="Train and Validation Errors vs Lambda",
                                     xaxis=dict(title="Lambda", type="log"),
                                     yaxis=dict(title="Error Value")))
    fig.write_html(os.path.join(output_dir, 'logistic_cross_validation_errors.html'))


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    fit_logistic_regression()

import numpy as np
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adaboost import AdaBoost
from decision_stump import DecisionStump

def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y

def plot_errors(train_errors, test_errors, noise):
    """
    Plot train and test errors vs number of learners and save the plot.

    Parameters
    ----------
    train_errors: list
        List of training errors for each number of learners

    test_errors: list
        List of testing errors for each number of learners

    noise: float
        Noise level for generating data
    """
    fig = make_subplots(rows=1, cols=1, subplot_titles=("Train and Test Errors vs Number of Learners"))
    fig.add_trace(go.Scatter(x=list(range(1, len(train_errors) + 1)), y=train_errors, mode='lines', name='Train Error'))
    fig.add_trace(go.Scatter(x=list(range(1, len(test_errors) + 1)), y=test_errors, mode='lines', name='Test Error'))
    fig.update_layout(title_text=f'Train and Test Errors vs Number of Learners (Noise={noise})',
                      xaxis_title='Number of Learners', yaxis_title='Error')
    fig.write_html(f"errors_noise_{noise}.html")

def plot_decision_boundaries(ada_boost, train_X, train_y, test_X, test_y, T, noise):
    """
    Plot decision boundaries for different numbers of learners and save the plot.

    Parameters
    ----------
    ada_boost: AdaBoost
        Trained AdaBoost ensemble

    train_X: np.ndarray
        Training samples

    train_y: np.ndarray
        Training labels

    test_X: np.ndarray
        Test samples

    test_y: np.ndarray
        Test labels

    T: list
        List of numbers of learners to use for plotting decision boundaries

    noise: float
        Noise level for generating data
    """
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=1, cols=4, subplot_titles=[str(num) + " Classifiers" for num in T])

    for i in range(len(T)):
        surf = decision_surface(lambda X: ada_boost.partial_predict(X, T[i]), lims[0], lims[1], density=60,
                                showscale=False)
        scat_dict = {"color": test_y, "symbol": np.where(test_y == 1, "circle", "x")}
        scat = go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False, marker=scat_dict)
        fig.add_traces([surf, scat], rows=1, cols=1 + i)

    fig.update_layout(title=f'Decision Boundaries (Noise={noise})', height=500, width=2000, margin={"t": 100})
    fig.write_html(f"decision_boundaries_noise_{noise}.html")

def plot_best_ensemble_decision_surface(ada_boost_best, train_X, train_y, test_X, test_y, best_t, best_test_error,
                                        noise):
    """
    Plot the decision surface of the best performing ensemble and save the plot.

    Parameters
    ----------
    ada_boost_best: AdaBoost
        Trained AdaBoost ensemble with the best performance

    train_X: np.ndarray
        Training samples

    train_y: np.ndarray
        Training labels

    test_X: np.ndarray
        Test samples

    test_y: np.ndarray
        Test labels

    best_t: int
        Number of learners for the best performing ensemble

    best_test_error: float
        Test error of the best performing ensemble

    noise: float
        Noise level for generating data
    """
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    accuracy = 1 - best_test_error
    fig = make_subplots(rows=1, cols=1, subplot_titles=[
        f'Best Ensemble (Size={best_t}, Test Error={best_test_error:.2f}, Accuracy={accuracy:.2f})'])

    surf = decision_surface(lambda X: ada_boost_best.partial_predict(X, best_t), lims[0], lims[1], density=60,
                            showscale=False)
    scat_dict = {"color": test_y, "symbol": np.where(test_y == 1, "circle", "x")}
    scat = go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False, marker=scat_dict)
    fig.add_traces([surf, scat], rows=1, cols=1)

    fig.update_layout(title="Best Performing Ensemble", height=700, width=700, margin={"t": 100})
    fig.write_html(f"best_ensemble_noise_{noise}.html")


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    """
    Train and evaluate AdaBoost ensemble and save relevant plots.

    Parameters
    ----------
    noise: float
        Noise level for generating data

    n_learners: int
        Number of weak learners

    train_size: int
        Number of training samples

    test_size: int
        Number of testing samples
    """
    # Generate train and test data
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Train AdaBoost ensemble
    ada_boost = AdaBoost(wl=DecisionStump, iterations=n_learners)
    ada_boost.fit(train_X, train_y)

    # Task 1: Plot train and test errors vs number of learners
    train_errors = []
    test_errors = []
    for t in range(1, n_learners + 1):
        train_error = ada_boost.partial_loss(train_X, train_y, T=t)
        test_error = ada_boost.partial_loss(test_X, test_y, T=t)
        train_errors.append(train_error)
        test_errors.append(test_error)

    plot_errors(train_errors, test_errors, noise)

    # Task 2: Plot decision boundaries for T = [5, 50, 100, 250]
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    plot_decision_boundaries(ada_boost, train_X, train_y, test_X, test_y, T, noise)

    # Task 3: Plot decision surface of best performing ensemble
    best_t = np.argmin(test_errors) + 1
    best_test_error = test_errors[best_t - 1]
    ada_boost_best = AdaBoost(wl=DecisionStump, iterations=best_t)
    ada_boost_best.fit(train_X, train_y)
    plot_best_ensemble_decision_surface(ada_boost_best, train_X, train_y, test_X, test_y, best_t, best_test_error, noise)

    # Task 4: Plot weighted training set and decision surface
    class_symbols = ["circle", "x"]  # Define the symbols for each class

    # Ensure D is normalized
    D_normalized = ada_boost.D_ / np.max(ada_boost.D_) * 5

    fig = go.Figure(layout=go.Layout(title="Final adaboost sample distribution",
                                     margin=dict(t=100)))

    fig.add_traces([decision_surface(ada_boost.predict, lims[0], lims[1], showscale=False),
            go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                       marker=dict(color=train_y,
                                   symbol=[class_symbols[int(label)] for label in train_y],
                                   size=D_normalized,
                                   line=dict(color="black", width=1)))])

    fig.update_layout(width=500, height=500)
    fig.write_html(f"final_adaboost_sample_distribution_noise_{noise}.html")

if __name__ == '__main__':
    np.random.seed(0)
    # Evaluate AdaBoost with noise level 0.0
    fit_and_evaluate_adaboost(noise=0.0)
    # Evaluate AdaBoost with noise level 0.4
    fit_and_evaluate_adaboost(noise=0.4)

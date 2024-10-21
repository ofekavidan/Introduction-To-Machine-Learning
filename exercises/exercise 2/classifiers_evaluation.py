from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import numpy as np

def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        data = np.load(f)
        X, y = data[:,:2], data[:,2]


        # Fit Perceptron and record loss in each fit iteration
        losses = []
        def collect_losses(model, cur_X, cur_y):
            losses.append(model.loss(X,y))

        new_model = Perceptron(callback=collect_losses).fit(X,y)

        # Plot figure of loss as function of fitting iteration
        plt.plot(range(len(losses)), losses, label=n)

        plt.xlabel('Iteration')
        plt.ylabel('Training Loss')
        plt.title('Perceptron Training Loss Over Iterations')
        plt.legend()
        plt.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """

    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")

def plot_ellipse(ax, mu, cov, color):
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * np.sqrt(2 * eigenvalues)
    ell = Ellipse(xy=mu, width=width, height=height, angle=angle, edgecolor=color, fc='None', lw=2)
    ax.add_patch(ell)

def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        data = np.load(f)
        X, y = data[:, :2], data[:, 2]

        # Fit models and predict over training set
        LDA_model = LDA().fit(X,y)
        naive_model = GaussianNaiveBayes().fit(X,y)
        predicted_LDA = LDA_model.predict(X)
        predicted_naive = naive_model.predict(X)

        # Calculate accuracies
        from loss_functions import accuracy
        lda_accuracy = accuracy(y, predicted_LDA)
        naive_accuracy = accuracy(y, predicted_naive)

        # Create figure and axes
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Subplot (a): Gaussian Naive Bayes predictions
        scatter_naive = axs[0].scatter(X[:, 0], X[:, 1], c=predicted_naive, cmap='coolwarm', edgecolor='k',
                                       label='Predicted')
        axs[0].scatter(naive_model.mu_[:, 0], naive_model.mu_[:, 1], marker='X', s=100, color='black',
                       label='Gaussian Centers')
        for mu, cov in zip(naive_model.mu_, naive_model.vars_):
            plot_ellipse(axs[0], mu, np.diag(cov), color='black')
        axs[0].set_title(f'Gaussian Naive Bayes\nAccuracy: {naive_accuracy:.2f}')
        axs[0].set_xlabel('Feature 1')
        axs[0].set_ylabel('Feature 2')
        axs[0].legend()

        # Subplot (b): LDA predictions
        scatter_lda = axs[1].scatter(X[:, 0], X[:, 1], c=predicted_LDA, cmap='coolwarm', edgecolor='k',
                                     label='Predicted')
        axs[1].scatter(LDA_model.mu_[:, 0], LDA_model.mu_[:, 1], marker='X', s=100, color='black',
                       label='Gaussian Centers')
        for mu, cov in zip(LDA_model.mu_, [LDA_model.cov_] * len(LDA_model.mu_)):
            plot_ellipse(axs[1], mu, cov, color='black')
        axs[1].set_title(f'LDA\nAccuracy: {lda_accuracy:.2f}')
        axs[1].set_xlabel('Feature 1')
        axs[1].set_ylabel('Feature 2')
        axs[1].legend()

        # Overall figure title
        fig.suptitle('Comparing Gaussian Classifiers on gaussians1.npy')

        # Display the plot
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()

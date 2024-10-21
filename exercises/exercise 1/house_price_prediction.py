from typing import NoReturn
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

from linear_regression import LinearRegression

def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """
    X = X[~(X['date'].isna())]
    X = preprocess_test(X)
    X.drop_duplicates(inplace=True)
    X = X[(X['yr_built'] <= X['yr_renovated']) | (X['yr_renovated'] == 0)]
    X = X[X['sqft_living'] > 400]
    X = X[X['bedrooms'] <= 15]

    y = y.loc[X.index]

    # filter on label
    y.dropna(inplace=True)
    X = X.loc[y.index]

    return X, y

def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    # dropped because we think are irrelevant to the price
    X = X.drop(["id", "date", "long", "lat"], axis=1)

    # dropped because of low Pearson Correlation
    X = X.drop(["condition"], axis=1)

    # todo: categorial column (idea: yr_built - split by decades)
    X['built_1900_1950'] = (X['yr_built'] < 1950).astype(int)
    X['built_1950_2000'] = ((X['yr_built'] >= 1950) & (X['yr_built'] < 2000)).astype(int)
    X['built_2000'] = (X['yr_built'] >= 2000).astype(int)

    return X

def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for feature in X.columns:
        feature_values = X[feature]
        covariance = np.cov(feature_values, y, bias=True)[0][1]
        std_feature = np.std(feature_values)
        std_response = np.std(y)
        pearson_corr = covariance / (std_feature * std_response)

        plt.figure(figsize=(8, 6))
        plt.scatter(feature_values, y, alpha=0.5)
        plt.title(f"{feature} - Pearson Correlation: {pearson_corr:.2f}")
        plt.xlabel(feature)
        plt.ylabel("Response")
        plt.grid(True)
        plt.savefig(output_path + "/" + feature + ".png")
        # plot_filename = os.path.join(output_path, f"{feature}_scatter.png")
        # plt.savefig(plot_filename)
        plt.close()

if __name__ == '__main__':
    if not os.path.exists("./feature_plots"):
        os.makedirs("./feature_plots")


    random_seed = 0
    df = pd.read_csv("house_prices.csv")
    X, y = df.drop("price", axis=1), df.price

    # Question 2 - split train test

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_seed)


    X_train =  X_train.reset_index(drop=True)
    y_train =  y_train.reset_index(drop=True)


    # Question 3 - preprocessing of housing prices train dataset
    X_train, y_train = preprocess_train(X_train, y_train)

    # Question 4 - Feature evaluation of train dataset with respect to response
    feature_evaluation(X_train, y_train, output_path="./feature_plots")
    #
    # Question 5 - preprocess the test data
    X_test = preprocess_test(X_test)

    # Question 6 - Fit model over increasing percentages of the overall training data


    num_iter = 10
    p_vals = list(range(10, 101, 1))
    p_mean_vals, p_std_vals = [], []
    for p in p_vals:
        loss_arr = []
        for i in range(num_iter):
            model = LinearRegression()
            X_sampled = X_train.sample(frac=p/100)
            y_sampled = y_train.loc[X_sampled.index]
            model.fit(X_sampled, y_sampled)
            loss_arr.append(model.loss(X_test, y_test))

        # calculate train metrics
        p_mean_vals.append(np.mean(loss_arr))
        p_std_vals.append(np.std(loss_arr))

    # Create the plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=p_vals,
        y=p_mean_vals,
        mode='lines+markers',
        name='Mean Loss',
        line=dict(color='royalblue', width=2),
        marker=dict(size=6)
    ))

    fig.add_trace(go.Scatter(
        x=p_vals,
        y=np.array(p_mean_vals) + 2 * np.array(p_std_vals),
        fill=None,
        mode='lines',
        line=dict(color='lightgrey', width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=p_vals,
        y=np.array(p_mean_vals) - 2 * np.array(p_std_vals),
        fill='tonexty',
        mode='lines',
        line=dict(color='lightgrey', width=0),
        name='Confidence Interval'
    ))

    fig.update_layout(
        title='Mean Loss and Confidence Interval as Function of Training Set Size',
        xaxis_title='Training Set Percentage (%)',
        yaxis_title='Mean Squared Error',
        template='plotly_white',
        width=800,
        height=500
    )

    fig.write_html('./feature_plots/Q6_houseprice.html')


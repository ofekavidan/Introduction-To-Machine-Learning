import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from polynomial_fitting import PolynomialFitting
import os


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # Load the dataset
    df = pd.read_csv(filename, parse_dates=['Date'])

    # # Handle invalid data by dropping rows with missing or malformed data
    # df.dropna(inplace=True)

    # Drop rows with invalid data (assuming Temp < -50 or Temp > 50 are invalid)
    df = df[(df['Temp'] > -50) & (df['Temp'] < 50)]

    # Add a 'DayOfYear' column based on the 'Date' column
    df['DayOfYear'] = df['Date'].dt.dayofyear

    return df

def plot_israel_temperature(df):
    # Filter dataset for Israel
    df_israel = df[df['Country'] == 'Israel']

    # Plot average daily temperature as a function of the DayOfYear
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_israel, x='DayOfYear', y='Temp', hue='Year', palette='tab10')
    plt.title('Average Daily Temperature in Israel as a Function of DayOfYear')
    plt.xlabel('Day of Year')
    plt.ylabel('Temperature (°C)')
    plt.legend(title='Year')
    plt.savefig("./feature_plots/AverageDailyTemperatureInIsrael.png")

    # Group by Month and calculate standard deviation of daily temperatures
    monthly_std = df_israel.groupby('Month')['Temp'].agg('std')

    # Plot the standard deviation for each month
    plt.figure(figsize=(10, 6))
    monthly_std.plot(kind='bar', color='skyblue')
    plt.title('Standard Deviation of Daily Temperatures by Month in Israel')
    plt.xlabel('Month')
    plt.ylabel('Standard Deviation (°C)')
    plt.savefig("./feature_plots/StandardDeviationDailyTemperatureInIsrael.png")


def plot_country_temperature(df):
    # Group by Country and Month, then calculate average and standard deviation
    grouped = df.groupby(['Country', 'Month']).agg({'Temp': ['mean', 'std']}).reset_index()
    grouped.columns = ['Country', 'Month', 'TempMean', 'TempStd']

    # Plot average monthly temperature with error bars
    fig = px.line(
        grouped,
        x='Month',
        y='TempMean',
        color='Country',
        error_y='TempStd',
        title='Average Monthly Temperature with Standard Deviation by Country',
        labels={'TempMean': 'Average Temperature (°C)', 'Month': 'Month'}
    )

    fig.update_layout(yaxis_title='Average Temperature (°C)', xaxis_title='Month')
    # fig.show()
    fig.write_html("./feature_plots/Q4_city_temp.html")

def fit_polynomial_models(df):
    # Filter data for Israel
    df = df[df['Country'] == 'Israel']

    X, y = df.drop("Temp", axis=1), df.Temp


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size=0.25, random_state=42)


    test_errors = []

    # Fit polynomial models of degree 1 to 10
    for k in range(1, 11):
        poly = PolynomialFitting(k)
        poly.fit(X_train.DayOfYear, y_train)
        test_error = poly.loss(X_test.DayOfYear, y_test)
        test_errors.append(round(test_error, 2))

        print(f'Degree {k}: Test Error = {test_errors[-1]}')

    # Plot test errors
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 11), test_errors, color='skyblue')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Test Error (MSE)')
    plt.title('Test Error for Polynomial Models of Different Degrees')
    plt.savefig("./feature_plots/TestErrorDiffDegrees.png")

    # Determine the best degree
    best_degree = np.argmin(test_errors) + 1
    # print(f'The best polynomial degree is {best_degree} with a test error of {test_errors[best_degree - 1]}')
    return best_degree

def evaluate_model_on_countries(df, best_degree):
    # Filter data for Israel
    israel_df = df[df['Country'] == 'Israel']

    # Extract features and target variable
    X_israel = israel_df['DayOfYear']
    y_israel = israel_df['Temp']

    # Fit polynomial model with the best degree on Israel data
    poly = PolynomialFitting(best_degree)
    X_israel_poly = poly.fit(X_israel, y_israel)


    # Evaluate model on other countries
    countries = df['Country'].unique()
    countries = countries[countries != 'Israel']

    errors = {}
    for country in countries:
        country_df = df[df['Country'] == country]
        X_country = country_df['DayOfYear']
        y_country = country_df['Temp']

        error = poly.loss(X_country, y_country)
        errors[country] = round(error, 2)

    # Plotting the errors
    plt.figure(figsize=(10, 6))
    plt.bar(errors.keys(), errors.values(), color='skyblue')
    plt.xlabel('Country')
    plt.ylabel('Test Error (MSE)')
    plt.title(f'Test Error for Model Fitted on Israel Data (Degree {best_degree})')
    plt.savefig("./feature_plots/TestErrorIsrael.png")

    return errors


if __name__ == '__main__':
    if not os.path.exists("./feature_plots"):
        os.makedirs("./feature_plots")

    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")
    # print(df.head())

    # Question 3 - Exploring data for specific country
    plot_israel_temperature(df)

    # Question 4 - Exploring differences between countries
    plot_country_temperature(df)

    # Question 5 - Fitting model for different values of `k`
    best_degree = fit_polynomial_models(df)

    # Question 6 - Evaluating fitted model on different countries
    country_errors = evaluate_model_on_countries(df, best_degree)
    # print(country_errors)
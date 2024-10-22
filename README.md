# Introduction to Machine Learning
**Hebrew University, Jerusalem, Israel**

An introductory code to the world of machine- and statistical learning, aimed for undergraduate students of computer science. The following repository contains:
1) Course Book - Based on lecture- and recitation notes
2) Code examples and graphs generating code, used throughout the book
3) Hands-on guided labs to experience aspects discussed in lectures and recitations
4) Skeleton files for software package ``IMLearn`` developed throughout the course
5) Skeleton files for course exercises


---

## 📚 Table of Contents

- [Exercise 1: House Price Prediction & Temperature Analysis](#exercise-1-house-price-prediction--temperature-analysis)
- [Exercise 2: Classifier Evaluation and Loss Functions](#exercise-2-classifier-evaluation-and-loss-functions)
- [Exercise 3: Boosting and AdaBoost](#exercise-3-boosting-and-adaboost)
- [Exercise 4: Logistic Regression and Gradient Descent Investigation](#exercise-4-logistic-regression-and-gradient-descent-investigation)


---

## Exercise 1: House Price Prediction & Temperature Analysis

In this exercise, linear and polynomial regression models were used to predict house prices and analyze city temperature data over time. The main objective was to understand how regression techniques can model real-world data.

- **Key Topics**: Linear Regression, Polynomial Fitting, Data Preprocessing.
- **Technologies**: Python, NumPy, Scikit-learn.

[More Details](https://github.com/ofekavidan/Introduction-To-Machine-Learning/tree/main/exercises/exercise%201)

---

## Exercise 2: Classifier Evaluation and Loss Functions

This exercise focused on implementing and evaluating several classifiers such as Naive Bayes, Logistic Regression, and SVM. The models were optimized using different loss functions and were evaluated using various performance metrics.

- **Key Topics**: Naive Bayes, Logistic Regression, SVM, Loss Functions.
- **Technologies**: Python, NumPy, Scikit-learn.

[More Details](https://github.com/ofekavidan/Introduction-To-Machine-Learning/tree/main/exercises/exercise%202)

---

## Exercise 3: Boosting and AdaBoost

In this exercise, the AdaBoost algorithm was implemented using weak classifiers like decision stumps. The objective was to explore the power of boosting techniques in building strong classifiers from weak learners.

- **Key Topics**: AdaBoost, Decision Stumps, Boosting Algorithms.
- **Technologies**: Python, NumPy, Scikit-learn.

[More Details](https://github.com/ofekavidan/Introduction-To-Machine-Learning/tree/main/exercises/exercise%203)

---

## Exercise 4: Logistic Regression and Gradient Descent Investigation

This exercise involved implementing logistic regression with gradient descent as the optimization method. An investigation into the impact of various learning rates on gradient descent convergence was conducted, with visualizations of the optimization process.

- **Key Topics**: Logistic Regression, Gradient Descent, Learning Rate Tuning.
- **Technologies**: Python, NumPy, Matplotlib.

[More Details](https://github.com/ofekavidan/Introduction-To-Machine-Learning/tree/main/exercises/exercise%204)

---



## Setting Up Code and Labs Environment
Set a local copy of this GitHub repository. Do so by [forking and cloning the repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo) or [cloning the repository](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository) using GitBash and 
```
cd LOCAL_REPOSITORY_PATH
git clone https://github.com/GreenGilad/IML.HUJI.git
```

or by downloading and unzipping it in `LOCAL_REPOSITORY_PATH`. Then:

### Anaconda + Jupyter Notebooks Setup
- Download and install Anaconda from [official website](https://www.anaconda.com/products/individual#Downloads). 
- Verify instllation by starting the Anaconda Prompt. A terminal should start with the text `(base)` written at the beginning of the line.
- Set the IML conda environment. Start the Anaconda Prompt and write:
  ```
  conda env create -f "LOCAL_REPOSITORY_PATH\environment.yml"
  ```
  This will create a conda envoronment named `iml.env` with the specifications defined in `environment.yml`. If creating failed due to `ResolvePackageNotFound: plotly-orca` remove this line from environment file, create environment without, and then after activating environment run:
  ```
  conda install -c plotly plotly-orca
  ```
- Activate the environment by `conda activate iml.env`.
- To open one of the Jupyter notebooks:
  ```
  jupyter notebook "LOCAL_REPOSITORY_PATH\lab\Lab 00 - A - Data Analysis In Python - First Steps.ipynb"
  ```

### Using PyCharm
Another option is to run the Jupyter notebooks through the PyCharm IDE plug-in. 
- Install the PyCharm IDE (professional edition) as described on the [Install PyCharm](https://www.jetbrains.com/help/pycharm/installation-guide.html) page. Be sure to install the Jupyter plug-in.
- Follow the [Configure a Conda virtual environment](https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html#conda-requirements) page.
- Open a PyCharm project from `LOCAL_REPOSITORY` and set the project's interpreter to be the `iml.env` environment.

### Using Google Colab
One can also view and run the labs and code examples via Google Colab. It supports loading and running Jupyter notebooks and running using a specified Conda environemnt.

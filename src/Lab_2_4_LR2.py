import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


class LinearRegressor:
    """
    Extended Linear Regression model with support for categorical variables and gradient descent fitting.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    """
    This next "fit" function is a general function that either calls the *fit_multiple* code that
    you wrote last week, or calls a new method, called *fit_gradient_descent*, not implemented (yet)
    """

    def fit(self, X, y, method="least_squares", learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): method to train linear regression coefficients.
                          It may be "least_squares" or "gradient_descent".
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if method not in ["least_squares", "gradient_descent"]:
            raise ValueError(
                f"Method {method} not available for training linear regression."
            )
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        X_with_bias = np.insert(
            X, 0, 1, axis=1
        )  # Adding a column of ones for intercept

        if method == "least_squares":
            self.fit_multiple(X_with_bias, y)
        elif method == "gradient_descent":
            self.fit_gradient_descent(X_with_bias, y, learning_rate, iterations)


    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression with the normal equation.
        
        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # Calcular los parámetros usando la ecuación normal: theta = (X^T X)^(-1) X^T y
        theta = np.linalg.inv(X.T @ X) @ X.T @ y

        # Guardar el intercepto (primer elemento) y los coeficientes (restantes)
        self.intercept = theta[0]
        self.coefficients = theta[1:]

    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000):
        """
        Fit the model using gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        m = len(y)  # Número de muestras
        
        # Inicializar los parámetros aleatoriamente con valores pequeños
        theta = np.random.rand(X.shape[1]) * 0.01

        # Bucle de descenso de gradiente
        for epoch in range(iterations):
            predictions = X @ theta      # Calcular las predicciones
            error = predictions - y      # Calcular el error

            # Calcular el gradiente: (X^T * error) / m
            gradient = (X.T @ error) / m

            # Actualizar los parámetros
            theta -= learning_rate * gradient

            # Imprimir el error cuadrático medio cada 100 iteraciones
            if epoch % 100 == 0:
                mse = np.mean(error ** 2)
                print(f"Época {epoch}: MSE = {mse}")

        # Guardar el intercepto y los coeficientes aprendidos
        self.intercept = theta[0]
        self.coefficients = theta[1:]


    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).

        Returns:
            np.ndarray: Predicted values of the dependent variable.
        """
        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")

        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        return X @ self.coefficients + self.intercept



def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """
    # Calcular R^2
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r_squared = 1 - (ss_residual / ss_total)

    # Calcular RMSE (Raíz del Error Cuadrático Medio)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Calcular MAE (Error Absoluto Medio)
    mae = np.mean(np.abs(y_true - y_pred))

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}


def one_hot_encode(X, categorical_indices, drop_first=False):
    """
    One-hot encode the categorical columns specified in categorical_indices. This function
    shall support string variables.

    Args:
        X (np.ndarray): 2D data array.
        categorical_indices (list of int): Indices of columns to be one-hot encoded.
        drop_first (bool): Whether to drop the first level of one-hot encoding to avoid multicollinearity.

    Returns:
        np.ndarray: Transformed array with one-hot encoded columns.
    """
    X_transformed = X.copy()

    # Procesar cada columna categórica (comenzando desde la última para evitar problemas de indexación)
    for index in sorted(categorical_indices, reverse=True):
        # Extraer la columna categórica
        categorical_column = X[:, index]
        # Obtener los valores únicos de la columna
        unique_values = np.unique(categorical_column)

        # Crear la matriz de codificación one-hot
        one_hot = np.array([categorical_column == category for category in unique_values]).T.astype(int)

        # Si se solicita eliminar la primera columna de la codificación
        if drop_first:
            one_hot = one_hot[:, 1:]

        # Eliminar la columna original y concatenar las columnas codificadas
        X_transformed = np.delete(X_transformed, index, axis=1)
        X_transformed = np.hstack((X_transformed, one_hot))

    return X_transformed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

def load_and_clean_dataset(dataset_path):
    # Load the dataset
    data = pd.read_csv('rainfall_area-wt_sd_1901-20151.csv', delimiter=',', encoding='latin-1')

    # Handle missing values only in the numeric columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    imputer = SimpleImputer(strategy='mean')
    data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

    return data

def split_data(data):
    # Split the dataset into features (X) and target (y)
    X = data.iloc[:, 2:-4]  # Exclude the first two columns (Subdivision and Year)
    y = data['Jun-Sep']  # Change this column name according to your target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def train_linear_regression(X_train, y_train):
    mlr = LinearRegression()
    mlr.fit(X_train, y_train)
    return mlr

def train_knn_regression(X_train, y_train):
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    return knn

def train_random_forest_regression(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

def plot_results(y_test, y_pred, model_name):
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{model_name} Predictions vs Actual")
    plt.show()

def main():
    # Load and clean the dataset
    dataset_path = "rainfall_area-wt_sd_1901-20151.csv"  # Replace with your dataset file path
    data = load_and_clean_dataset(dataset_path)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(data)

    # Train models
    mlr = train_linear_regression(X_train, y_train)
    knn = train_knn_regression(X_train, y_train)
    rf = train_random_forest_regression(X_train, y_train)

    # Make predictions
    mlr_predictions = mlr.predict(X_test)
    knn_predictions = knn.predict(X_test)
    rf_predictions = rf.predict(X_test)

    # Calculate accuracy metrics
    mlr_mse, mlr_r2 = calculate_metrics(y_test, mlr_predictions)
    knn_mse, knn_r2 = calculate_metrics(y_test, knn_predictions)
    rf_mse, rf_r2 = calculate_metrics(y_test, rf_predictions)

    # Print the results
    print("Multiple Linear Regression:")
    print(f"Mean Squared Error: {mlr_mse}")
    print(f"R-squared: {mlr_r2}")
    print("\nK-Nearest Neighbors:")
    print(f"Mean Squared Error: {knn_mse}")
    print(f"R-squared: {knn_r2}")
    print("\nRandom Forest:")
    print(f"Mean Squared Error: {rf_mse}")
    print(f"R-squared: {rf_r2}")

    # Plot results
    plot_results(y_test, mlr_predictions, "Multiple Linear Regression")
    plot_results(y_test, knn_predictions, "K-Nearest Neighbors")
    plot_results(y_test, rf_predictions, "Random Forest")

    # Select the best model based on metrics
    best_model = None
    best_mse = float('inf')
    best_r2 = -float('inf')

    if mlr_mse < best_mse and mlr_r2 > best_r2:
        best_model = "Multiple Linear Regression"
        best_mse = mlr_mse
        best_r2 = mlr_r2

    if knn_mse < best_mse and knn_r2 > best_r2:
        best_model = "K-Nearest Neighbors"
        best_mse = knn_mse
        best_r2 = knn_r2

    if rf_mse < best_mse and rf_r2 > best_r2:
        best_model = "Random Forest"
        best_mse = rf_mse
        best_r2 = rf_r2

    print(f"\nBest Model: {best_model}")
    print(f"Best Mean Squared Error: {best_mse}")
    print(f"Best R-squared: {best_r2}")

if __name__ == "__main__":
    main()

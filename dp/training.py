import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import joblib

# Load your dataset (replace 'your_dataset.csv' with your actual file path)
data = pd.read_csv('rainfall_area-wt_sd_1901-20151.csv', delimiter=',', encoding='latin-1')
data.dropna(inplace=True)
# Handle missing values only in the numeric columns
numeric_columns = data.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='mean')
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

# Preprocess 'LATITUDE' and 'LONGITUDE' columns to extract numerical values
data['LATITUDE'] = data['LATITUDE'].str.replace('°N', '').astype(float)
data['LONGITUDE'] = data['LONGITUDE'].str.replace('°E', '').astype(float)

# Select the 'LATITUDE' and 'LONGITUDE' columns as input features
X = data[['LATITUDE', 'LONGITUDE']]
# Define your target variable (replace 'Jun-Sep' with your actual target column name)
y = data['Jun-Sep']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a multiple linear regression model
mlr = LinearRegression()
mlr.fit(X_train, y_train)

# Save the trained model to a .pkl file
model_filename = "trained_mlr_model.pkl"
joblib.dump(mlr, model_filename)
print(f"Trained multiple linear regression model saved as {model_filename}")

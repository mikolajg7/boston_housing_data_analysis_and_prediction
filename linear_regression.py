from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Load the dataset
file_path = 'hou_all.csv'
headers = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV','dtype']
data = pd.read_csv(file_path, header=None, names=headers)

# Preprocess the data
# Assuming 'MEDV' is the target variable (median value of owner-occupied homes)
X = data.drop(['MEDV', 'dtype'], axis=1)  # Exclude 'dtype'
y = data['MEDV']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Evaluate the model on the testing data
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# Define an example house
example_house = {
    'CRIM': 0.00001,
    'ZN': 18.0,
    'INDUS': 2.31,
    'CHAS': 0,
    'NOX': 0.538,
    'RM': 6.575,
    'AGE': 65.2,
    'DIS': 4.09,
    'RAD': 1,
    'TAX': 296,
    'PTRATIO': 15.3,
    'B': 396.9,
    'LSTAT': 4.98,
    'dtype': 'Residential'
}

# Convert the dictionary to a pandas DataFrame
example_house_df = pd.DataFrame([example_house])

# Use the model to make a prediction
predicted_price = model.predict(example_house_df.drop('dtype', axis=1))
print(f"Predicted price: {predicted_price[0]}")

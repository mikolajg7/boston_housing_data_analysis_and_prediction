import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to the CSV file
file_path = 'hou_all.csv'

# Define the headers for the data
headers = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV','dtype']


# CRIM (Per Capita Crime Rate by Town): The number of crimes per capita.
# ZN (Proportion of Residential Land Zoned for Lots Over 25,000 sq.ft.): The proportion of residential land for large lots.
# INDUS (Proportion of Non-Retail Business Acres per Town): The share of non-retail business acres per town.
# CHAS (Charles River Dummy Variable): A binary variable indicating proximity to the Charles River.
# NOX (Nitric Oxides Concentration): The concentration of nitric oxides in the air.
# RM (Average Number of Rooms per Dwelling): The average number of rooms per dwelling.
# AGE (Proportion of Owner-Occupied Units Built Prior to 1940): The proportion of owner-occupied units built before 1940.
# DIS (Weighted Distances to Five Boston Employment Centres): The weighted average distance to five Boston employment centers.
# RAD (Index of Accessibility to Radial Highways): The index of accessibility to radial highways.
# TAX (Full-Value Property-Tax Rate per $10,000): The property tax rate per $10,000 of full value.
# PTRATIO (Pupil-Teacher Ratio by Town): The ratio of pupils to tea chers in the town.
# B (1000(Bk - 0.63)^2 where Bk is the Proportion of Blacks by Town): An expression related to the proportion of Black population.
# LSTAT (% Lower Status of the Population): The percentage of the population with lower socioeconomic status.


# Read the CSV file into a pandas DataFrame
data=pd.read_csv(file_path,header=None,names=headers)

# Print basic statistics of the data
print(data.head())
print(data.info())
print(data.describe())

# Check for missing data
print("Missing data:")
print(data.isnull().sum())

# Check for duplicates
print("Number of duplicates:", data.duplicated().sum())

# Data visualization
sns.set(style="whitegrid", context="notebook")
plt.figure(figsize=(12, 8))

# Visualize the distribution of each feature in the dataset
data.hist(bins=20, figsize=(16, 10))
plt.show()

# Visualize the correlation matrix using a heatmap
numeric_columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
correlation_matrix = data[numeric_columns].corr()
plt.figure(figsize=(13, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.show()

# Visualize the relationship between each feature and the target variable
selected_features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
for feature in selected_features:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[feature], y=data['MEDV'])
    plt.title(f"{feature} vs MEDV")
    plt.xlabel(feature)
    plt.ylabel("MEDV")
    plt.show()

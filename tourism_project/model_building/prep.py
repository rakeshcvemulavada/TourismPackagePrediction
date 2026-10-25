# ==========================================
# Data Preprocessing Script for Tourism Project
# ==========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Load the raw dataset
df = pd.read_csv("tourism_project/data/tourism.csv")

# 2. Drop the unnamed index column if it exists to clean up the dataframe
if 'Unnamed: 0' in df.columns or df.columns[0] == '':
    df = df.iloc[:, 1:]

# 3. Drop CustomerID as it is a unique identifier and has no predictive power
if 'CustomerID' in df.columns:
    df.drop(columns=['CustomerID'], inplace=True)

# 4. Handle missing values systematically
print("\nHandling missing values...")
# For numerical columns, impute missing entries with the median value
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# For categorical columns, impute missing entries with the mode (most frequent value)
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# 5. Clean inconsistent data entries
df['Gender'] = df['Gender'].str.strip().replace({'Fe Male': 'Female', 'Fe male': 'Female'})

# Combine 'Unmarried' with 'Single' for category unification
df["MaritalStatus"] = df["MaritalStatus"].str.strip().replace("Unmarried", "Single")

# 6. Split features (X) and target variable (y)
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# 7. Stratified Train-Test split to preserve class imbalance ratio
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Export clean split data to CSVs
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)

print("Type values kept as:", sorted(y.unique()))
print("Data prepared successfully: train/test splits written locally.")

"""Categorical Data Preprocessing."""

# Importing the necessary libraries
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Importing the dataset
dataset = pd.read_csv("dataset.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Handling missing data
imputer = Imputer(missing_values = "NaN",
                  strategy = "mean",
                  axis = 0)

imputer.fit(X[:, 1:3])

X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding Categorical Data - Independent Variable
labelencoder_X = LabelEncoder()

X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(categorical_features = [0])

X = onehotencoder.fit_transform(X).toarray()

# Encoding Categorical Data - Dependent Variable
labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)

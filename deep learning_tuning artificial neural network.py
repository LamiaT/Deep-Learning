"""Tuning Artificial Neural Network."""

# Part 1 - Data Preprocessing
# Importing the necessary libraries for data preprocessing
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importing the dataset
dataset = pd.read_csv("dataset.csv")
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2 - Building ANN
# Importing necessary libraries with packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.metrics import confusion_matrix

# Initialising ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6,
                     kernel_initializer = "uniform",
                     activation = "relu",
                     input_dim = 11))

classifier.add(Dropout(p = 0.1))

# Adding the second hidden layer
classifier.add(Dense(units = 6,
                     kernel_initializer = "uniform",
                     activation = "relu"))

classifier.add(Dropout(p = 0.1))

# Adding the output layer
classifier.add(Dense(units = 1,
                     kernel_initializer = "uniform",
                     activation = "sigmoid"))

# Compiling ANN
classifier.compile(optimizer = "adam",
                   loss = "binary_crossentropy",
                   metrics = ["accuracy"])

# Fitting the ANN to the Training set
classifier.fit(X_train,
               y_train,
               batch_size = 10,
               epochs = 100)


# Part 3 - Making predictions using ANN
# Predicting Test Set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Predicting a single new observation
new_prediction = classifier.predict(sc.transform(np.array([[1, 1, 1, 1, 1,
                                                            1, 1, 1, 1, 1,
                                                            1]])))

new_prediction = (new_prediction > 0.5)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)


# Part 4 - Evaluating, Tuning and Improving ANN
# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


def classifier_builder():

    classifier = Sequential()
    classifier.add(Dense(units = 6, 
                         kernel_initializer = "uniform",
                         activation = "relu",
                         input_dim = 11))

    classifier.add(Dense(units = 6,
                         kernel_initializer = "uniform",
                         activation = "relu"))

    classifier.add(Dense(units = 1,
                         kernel_initializer = "uniform",
                         activation = "sigmoid"))

    classifier.compile(optimizer = "adam",
                       loss = "binary_crossentropy",
                       metrics = ["accuracy"])

    return classifier


classifier = KerasClassifier(build_fn = classifier_builder,
                             batch_size = 10,
                             epochs = 100)

accuracies = cross_val_score(estimator = classifier,
                             X = X_train,
                             y = y_train,
                             cv = 10,
                             n_jobs = -1)

mean = accuracies.mean()

variance = accuracies.std()

# Improving ANN
# Tuning ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


def classifier_builder(optimizer):

    classifier = Sequential()

    classifier.add(Dense(units = 6,
                         kernel_initializer = "uniform",
                         activation = "relu",
                         input_dim = 11))

    classifier.add(Dense(units = 6,
                         kernel_initializer = "uniform",
                         activation = "relu"))

    classifier.add(Dense(units = 1,
                         kernel_initializer = "uniform",
                         activation = "sigmoid"))

    classifier.compile(optimizer = optimizer,
                       loss = "binary_crossentropy",
                       metrics = ["accuracy"])

    return classifier


classifier = KerasClassifier(build_fn = classifier_builder)

parameters = {"batch_size": [25, 32],
              "epochs": [100, 500],
              "optimizer": ["adam", "rmsprop"]}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

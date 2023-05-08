# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Bank_Predictions.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# ------ Part-1: Data preprocessing ----------

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder_obj = ColumnTransformer([('Location_Encoder_Object', OneHotEncoder(), [1])], remainder='passthrough')
X = onehotencoder_obj.fit_transform(X)

X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# ------- Part-2: Build the ANN --------

# import keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()


# Adding the input layer and the first hidden layer
# The missing line of code here....
classifier.add(Dense(units=9, activation='relu', input_dim=11))

# Adding second hidden layer
# The missing line of code here....
classifier.add(Dense(units=9, activation='relu'))

# Adding output layer
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred)
y_pred = (y_pred > 0.5)
print(y_pred)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix of the mode is\n", cm)
print("Model Accuracy of ANN is", accuracy_score(y_test, y_pred))


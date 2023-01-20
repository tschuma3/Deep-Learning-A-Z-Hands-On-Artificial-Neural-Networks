import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

"""
This neural network predicts why a bank is loosing customers and if a future customer will leave
"""

#region Data Preprocessing

#Create variables and organize data
dataset = pd.read_csv(r'D:\GitHub Repos\Deep-Learning-A-Z-Hands-On-Artificial-Neural-Networks\Artificial Neural Network (ANN)\Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

#Encoding the Gender column
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

#Hot encoding Geography column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#endregion

#region Building the ANN

#Initialize the ANN
ann = tf.keras.models.Sequential()

#Adds the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))

#Adds hte second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))

#Adds the output layer
ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

#endregion

#region Training the ANN

#Complies ANN
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#Trains the ANN on the training set
ann.fit(X_train, y_train, batch_size=32, epochs=100)

#endregion

#region Making the Predictions and Evaluate

#Prints out a prediction
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

#Predicting the test results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#Making the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

#endregion
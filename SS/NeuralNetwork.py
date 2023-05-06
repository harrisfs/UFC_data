import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("UFC.csv")
dataset2 = pd.read_csv("UFCpred.csv")

# Define the data
X = dataset.iloc[:, range(1,62)].values
y = dataset.iloc[:, 0].values

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Define the prediction data
X_pred = dataset2.iloc[:, range(1,62)].values

import numpy as np

# Check for nan and inf values in X_train
print(np.isnan(X_train).any())
print(np.isinf(X_train).any())

# Check for nan and inf values in X_val
print(np.isnan(X_val).any())
print(np.isinf(X_val).any())

# Check for nan and inf values in X_pred
print(np.isnan(X_pred).any())
print(np.isinf(X_pred).any())

# Create the model
model = Sequential()
model.add(Dense(32, input_dim=61, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with accuracy metric
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the training data
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, verbose=0)

# Use the trained model to make predictions on the prediction data
predictions = model.predict(X_pred)
rounded = np.round(predictions)

# Print the rounded predictions
print(predictions)

# Evaluate the model on the validation data
val_loss = history.history['val_loss'][-1]
val_accuracy = history.history['val_accuracy'][-1]
print('Validation loss:', val_loss)
print('Validation accuracy:', val_accuracy)
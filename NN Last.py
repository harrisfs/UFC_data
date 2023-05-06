import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('UFC.csv')
datapred = pd.read_csv('UFCpred.csv')

# Split the data into X and y
X_train = data.iloc[:, 1:] # Independent variables for training
y_train = data.iloc[:, 0]  # Dependent variable for training
X_val = datapred.iloc[:, 1:] # Independent variables for validation
y_val = datapred.iloc[:, 0]


# Define the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=3000, batch_size=1024, validation_data=(X_val, y_val))

# Make predictions on the new dataset
new_probabilities = model.predict(X_val)
print("Predictions on new data:", new_probabilities[:,0]) # probabilities of player 1 winning

# Save the predicted probabilities to a CSV file
np.savetxt('my_array.csv', new_probabilities, delimiter=',')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('UFC.csv')
datapred = pd.read_csv('UFCpred.csv')

# Split the data into X and y
X = data.iloc[:, 1:] # Independent variables
y = data.iloc[:, 0]  # Dependent variable

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.16, shuffle=False)

# Create a logistic regression object
logreg = LogisticRegression()

# Fit the model on the training data
logreg.fit(X_train, y_train)

# Print the p-values of the logistic regression model
import statsmodels.api as sm
X2 = sm.add_constant(X_train)
est = sm.Logit(y_train, X2)
est2 = est.fit()
print(est2.summary())

# Make predictions on the test data
y_pred = logreg.predict(X_test)

# Print the accuracy score of the model
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))

# Print the predicted probabilities of the test data
probabilities = logreg.predict_proba(X_test)
print("Predictions:", probabilities[:,1]) # probabilities of player 1 winning

# Scale the new data
Xpred_scaled = scaler.transform(datapred.iloc[:, 1:])

# Make predictions on the new dataset
new_probabilities = logreg.predict_proba(Xpred_scaled)
print("Predictions on new data:", new_probabilities[:,1]) # probabilities of player 1 winning

np.savetxt('my_array.csv', new_probabilities, delimiter=',')

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the data
data = pd.read_csv('UFC.csv')
datapred = pd.read_csv('UFCpred.csv')

# Split the data into X and y
X = data.iloc[:, 1:] # Independent variables
y = data.iloc[:, 0]  # Dependent variable

# Add a constant to the independent variables
X = sm.add_constant(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a logistic regression object
logreg = sm.Logit(y_train, X_train).fit()

# Print the summary of the logistic regression model
print(logreg.summary())

# Select significant independent variables based on p-values
significant_vars = logreg.pvalues[logreg.pvalues < 0.05].index.values.tolist()
X_train_sig = X_train[significant_vars]
X_test_sig = X_test[significant_vars]
Xpred_sig = sm.add_constant(Xpred)[significant_vars]

# Create a new logistic regression object with significant variables only
logreg_sig = LogisticRegression()
logreg_sig.fit(X_train_sig, y_train)

# Make predictions on the test data
y_pred = logreg_sig.predict(X_test_sig)

# Print the accuracy score of the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Print the predicted probabilities of the test data
probabilities = logreg_sig.predict_proba(X_test_sig)
print("Predictions:", probabilities[:,1]) # probabilities of player 1 winning

# Make predictions on the new dataset with significant variables only
new_probabilities = logreg_sig.predict_proba(Xpred_sig)
print("Predictions on new data:", new_probabilities[:,1]) # probabilities of player 1 winning

# Save the predicted probabilities to a CSV file
np.savetxt('my_array.csv', new_probabilities, delimiter=',')


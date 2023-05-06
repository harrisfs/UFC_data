import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

dataset = pd.read_csv("UFC.csv")
dataset2= pd.read_csv("UFCpred.csv")


# input
x = dataset.iloc[:, range(1,63)].values
xt = dataset2.iloc[:, range(1,63)].values


# output
y = dataset.iloc[:, 0].values



from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.1, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain) 
xtest = sc_x.transform(xtest)
xt = sc_x.transform(xt)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='liblinear', random_state=0)
classifier.fit(xtrain, ytrain) 

y_pred = classifier.predict(xtest)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, y_pred)
  
print ("Confusion Matrix : \n", cm)


from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(ytest, y_pred))

y_pred2 = classifier.predict_proba(xt)

print(y_pred2)

print(classifier.coef_, classifier.intercept_)

import statsmodels.api as sm
logit_model=sm.Logit(ytrain,xtrain)
result=logit_model.fit()
print(result.summary())

# pred = np.fliplr(y_pred2)
# odds = dataset2.iloc[range(0,len(dataset2)), range(0,2)]
# prob = 1/odds
# value = (pred - prob)/prob
# b = odds -1 
# f = (pred - (1 - pred)/b)*20/0.3
# names = dataset2.iloc[range(0,len(dataset2)), range(2,4)]

# bets = pd.concat([names, odds, value,f], axis=1)
# #print(classifier.coef_, classifier.intercept_)

# np.savetxt("Output-bets.csv", bets, delimiter=",", fmt="%s")

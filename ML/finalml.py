import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Perform EDA

data = pd.read_csv('/Users/shiv/Downloads/assessment/data.csv')
print('Shape of the data:', data.shape)
print(data.info())
print('Number of missing values in each column:\n', data.isnull().sum())
print(data.describe())
data.hist(figsize=(10,10))
plt.show()
data.plot(kind='box', subplots=True, layout=(2,3), sharex=False, sharey=False, figsize=(10,10))
plt.show()
pd.plotting.scatter_matrix(data, figsize=(10,10))
plt.show()

#  Preprocess the data

X = data.iloc[:, :-1].values # Features
y = data.iloc[:, -1].values # Target
sc = StandardScaler()
X = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Train a logistic regression algorithm
clf = LogisticRegression(multi_class='ovr', max_iter=1000)
clf.fit(X_train, y_train)

#  Evaluate the trained model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
print(classification_report(y_test, y_pred))

#   Save the model
import pickle
filename = 'ta_evaluation_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(clf, file)

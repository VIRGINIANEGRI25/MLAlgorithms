import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt


col_names = ['tandelta', 'temperature', 'current', 'time', 'label']
# load dataset
pima = pd.read_csv('C:/Users/Virginia/Desktop/MLAlgorithms/dataset.csv', header=None, names=col_names)
print(pima.head())
#split dataset in features and target variable
feature_cols = ['tandelta', 'temperature', 'current', 'time']
X = pima[feature_cols] # Features
y = pima.label # Target variable

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# instantiate the model 
logreg = LogisticRegression(solver='lbfgs', max_iter=200)

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 5))
cmp = ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred),
    display_labels=["Class 0", "Class 1"],
)

cmp.plot(ax=ax)
plt.show();

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

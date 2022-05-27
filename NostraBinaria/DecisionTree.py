import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
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

# training a DescisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 3).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 5))
cmp = ConfusionMatrixDisplay(
    confusion_matrix(y_test, dtree_predictions),
    display_labels=["Yes", "No"],
)

cmp.plot(ax=ax)
plt.show();

print("Accuracy:",metrics.accuracy_score(y_test, dtree_predictions))

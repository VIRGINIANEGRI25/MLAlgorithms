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

# training a linear SVM
from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test)

# training a linear SVM classifier with ONE VS. ONE
from sklearn import svm
svm2 = svm.SVC(decision_function_shape="ovo").fit(X_train, y_train)
svm2_predictions = svm2.predict(X_test)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 5))
cmp = ConfusionMatrixDisplay(
    confusion_matrix(y_test, svm_predictions),
    display_labels=["Yes", "No"],
)

cmp.plot(ax=ax)
plt.show();

print("Accuracy:",metrics.accuracy_score(y_test, svm_predictions))
print("Accuracy:",metrics.accuracy_score(y_test, svm2_predictions))

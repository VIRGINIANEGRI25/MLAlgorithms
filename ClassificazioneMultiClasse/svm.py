from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics


col_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
# load dataset
pima = pd.read_csv('C:/Users/Virginia/Desktop/Algoritmi/ClassificazioneMultiClasse/iris.csv', header=None, names=col_names)
print(pima.head())
#split dataset in features and target variable
feature_cols = ['sepal length', 'sepal width', 'petal length', 'petal width']
X = pima[feature_cols] # Features
y = pima.label # Target variable

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
 
# training a linear SVM
from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test)

# training a linear SVM classifier with ONE VS. ONE
from sklearn import svm
svm2 = svm.SVC(decision_function_shape="ovo").fit(X_train, y_train)
svm2_predictions = svm2.predict(X_test)
round(svm2.score(X_test, y_test), 4)


# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 5))
cmp = ConfusionMatrixDisplay(
    confusion_matrix(y_test, svm_predictions),
    display_labels=["Iris Setosa", "Iris Versicolour", "Iris Virginica"],
)

cmp.plot(ax=ax)
plt.show();

print("Accuracy SVC:",metrics.accuracy_score(y_test, svm_predictions))
print("Accuracy SVM OVO:",metrics.accuracy_score(y_test, svm2_predictions))


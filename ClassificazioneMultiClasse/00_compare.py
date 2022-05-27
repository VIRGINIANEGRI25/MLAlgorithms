from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics


col_names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
# load dataset
pima = pd.read_csv('C:/Users/Virginia/Desktop/MLAlgorithms/ClassificazioneMultiClasse/iris.csv', header=None, names=col_names)
print(pima.head())
#split dataset in features and target variable
feature_cols = ['sepal length', 'sepal width', 'petal length', 'petal width']
X = pima[feature_cols] # Features
y = pima.label # Target variable

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
 

# training a Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(X_train, y_train)
rf_predictions = RF.predict(X_test)
print("Random Forest Accuracy:",metrics.accuracy_score(y_test, rf_predictions))

# training a Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(X_train, y_train)
gnb_predictions = gnb.predict(X_test)
print("Naive Bayes Accuracy:",metrics.accuracy_score(y_test, gnb_predictions))

# training a KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train)
knn_predictions = knn.predict(X_test)
print("k nearest neighbor Accuracy:",metrics.accuracy_score(y_test, knn_predictions))

# training a DescisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 3).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)
print("Decison Tree Accuracy:",metrics.accuracy_score(y_test, dtree_predictions))

# training a Neural Network classifier
from sklearn.neural_network import MLPClassifier
NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1).fit(X_train, y_train)
nn_predictions = NN.predict(X_test)
print("NN Accuracy:",metrics.accuracy_score(y_test, nn_predictions))

# training a linear SVM classifier
from sklearn.svm import SVC
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train)
svm_predictions = svm_model_linear.predict(X_test)
print("SVC Accuracy:",metrics.accuracy_score(y_test, svm_predictions))




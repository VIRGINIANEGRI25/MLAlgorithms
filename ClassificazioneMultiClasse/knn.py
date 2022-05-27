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
 
# training a KNN classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train)
knn_predictions = knn.predict(X_test)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(8, 5))
cmp = ConfusionMatrixDisplay(
    confusion_matrix(y_test, knn_predictions),
    display_labels=["Iris Setosa", "Iris Versicolour", "Iris Virginica"],
)

cmp.plot(ax=ax)
plt.show();

print("Accuracy:",metrics.accuracy_score(y_test, knn_predictions))
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import csv

def main():
    col_names = ['tandelta', 'temperature', 'current', 'time', 'label']
    # load dataset
    pima = pd.read_csv('C:/Users/Virginia/Desktop/MLAlgorithms/NostraBinaria/dataset.csv', header=None, names=col_names)
    print(pima.head())
    #split dataset in features and target variable
    feature_cols = ['tandelta', 'temperature', 'current', 'time']
    X = pima[feature_cols] # Features
    y = pima.label # Target variableknn.

    # dividing X, y into train and test data
    # n_test fixed at 100k
    n_test = 100000
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test, random_state=0, shuffle=False)


    # training a Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    myclassifier = RandomForestClassifier(random_state=0).fit(X_train, y_train)
    y_pred = myclassifier.predict(X_test)
    '''
    # per costruire matrice di confusione a parte
    with open('C:/Users/Virginia/Desktop/MLAlgorithms/NostraBinaria/ConfusionMatrix/actual.csv','w',newline='') as file:
        writer = csv.writer(file)
        writer.writerows(map(lambda x: [x], y_test))
    # per costruire matrice di confusione a parte
    with open('C:/Users/Virginia/Desktop/MLAlgorithms/NostraBinaria/ConfusionMatrix/predicted.csv','w',newline='') as file:
        writer = csv.writer(file)
        writer.writerows(map(lambda x: [x], y_pred))
    '''

    '''
    # Plot confusion matrix
    cmp = ConfusionMatrixDisplay.from_estimator(myclassifier,X_test,y_test,values_format='d',
        display_labels=["Class 0", "Class 1"],)
    cmp.plot
    plt.show();
    '''

    accurancy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)

    print("Accuracy:",accurancy)
    print("Precision:",precision)
    print("Recall:",recall)

  
    return accurancy


if __name__ == '__main__':
    main()

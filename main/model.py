import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


def plot_confusion_matrix(y_true, y_pred):
    ax = plt.subplot()
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.show()


def evaluate_model(model, X_train, Y_train, X_test, Y_test, param_grid=None, cv_folds=10):
    if param_grid:
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv_folds, scoring='accuracy')
        grid_search.fit(X_train, Y_train)
        best_model = grid_search.best_estimator_
        print(f"Best parameters for {model.__class__.__name__}: {grid_search.best_params_}")
        print(f"Train accuracy for {model.__class__.__name__}: {grid_search.best_score_}")
    else:
        best_model = model
        best_model.fit(X_train, Y_train)

    cv_scores = cross_val_score(best_model, X_train, Y_train, cv=cv_folds, scoring='accuracy')
    print(f"Cross-validation scores for {model.__class__.__name__}: {cv_scores}")
    print(f"Mean cross-validation accuracy: {cv_scores.mean():.4f}")

    test_accuracy = best_model.score(X_test, Y_test)
    print(f"Test accuracy for {model.__class__.__name__}: {test_accuracy:.4f}")

    y_pred = best_model.predict(X_test)
    plot_confusion_matrix(Y_test, y_pred)

    return best_model


data = pd.read_csv("Data/dataset_part_2.csv")
X = pd.read_csv("Data/dataset_part_3.csv")
Y = data['Class'].to_numpy()

# Standard Scaler
X = preprocessing.StandardScaler().fit_transform(X)

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Logistic Regression
lr_params = {
    'C': [0.01, 0.1, 1],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}
lr = LogisticRegression(max_iter=10000)
evaluate_model(lr, X_train, Y_train, X_test, Y_test, param_grid=lr_params)

# SVM
svm_params = {
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'C': np.logspace(-3, 3, 5),
    'gamma': np.logspace(-3, 3, 5)
}
svm = SVC()
evaluate_model(svm, X_train, Y_train, X_test, Y_test, param_grid=svm_params)

# Decision Tree
tree_params = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [2 * n for n in range(1, 10)],
    'max_features': ['sqrt', 'log2', None],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10]
}
tree = DecisionTreeClassifier()
evaluate_model(tree, X_train, Y_train, X_test, Y_test, param_grid=tree_params)

# KNN
knn_params = {
    'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2]
}
knn = KNeighborsClassifier()
evaluate_model(knn, X_train, Y_train, X_test, Y_test, param_grid=knn_params)

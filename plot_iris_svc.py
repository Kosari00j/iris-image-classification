
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd 

iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target


def linear_kernel(x, y):
    return np.dot(x, y.T)


def polynomial_kernel(x, y, C=1, degree=2):
    return (np.dot(x, y.T) + C) ** degree


def sigmoid_kernel(x, y, gamma=0.1, C=0.0):
    return np.tanh(gamma * np.dot(x, y.T) + C)


def gaussian_kernel(x, y, sigma=1):
    distances = np.sum((x[:, np.newaxis] - y) ** 2, axis=-1)
    kernel_matrix = np.exp(-distances / (2 * sigma ** 2))
    return kernel_matrix



# def linear_kernel(x, y):
#     result = np.dot(x, y.T)
#     print("Linear Kernel Result:", result)
#     return result

# def polynomial_kernel(x, y, C=1, degree=2):
#     result = (np.dot(x, y.T) + C) ** degree
#     print("Polynomial Kernel Result:", result)
#     return result

# def sigmoid_kernel(x, y, gamma=0.1, C=0.0):
#     result = np.tanh(gamma * np.dot(x, y.T) + C)
#     print("Sigmoid Kernel Result:", result)
#     return result

# def gaussian_kernel(x, y, sigma=1):
#     distances = np.sum((x[:, np.newaxis] - y) ** 2, axis=-1)
#     kernel_matrix = np.exp(-distances / (2 * sigma ** 2))
#     print("Gaussian Kernel Result:", kernel_matrix)
#     return kernel_matrix

C = 1.0
models = (
    svm.SVC(kernel=linear_kernel),
    svm.SVC(kernel=sigmoid_kernel, gamma=0.1, C=C),
    svm.SVC(kernel=gaussian_kernel, gamma=0.7, C=C),
    svm.SVC(kernel=polynomial_kernel, degree=2, C=C),
)



titles = (
    "SVC with linear kernel",
    "SVC with sigmoid kernel",
    "SVC with RBF kernel",
    "SVC with polynomial (degree 2) kernel",
)

fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]

for clf, title, ax in zip(models, titles, sub.flatten()):
    clf.fit(X, y)
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        ax=ax,
        xlabel=iris.feature_names[0],
        ylabel=iris.feature_names[1],
    )
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(f"{title}\nAccuracy: {accuracy:.2f}")
    print(f"{title} Accuracy: {accuracy:.2f}")

plt.show()




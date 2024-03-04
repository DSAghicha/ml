import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("p2a.csv")

print(data.head())
print(data.shape)

x = data.iloc[:, 0:13].values
y = data.iloc[:, 13].values

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=0)

sc = StandardScaler()
xTrain = sc.fit_transform(xTrain)
xTest = sc.transform(xTest)

pca = PCA(n_components=2)
xTrain = pca.fit_transform(xTrain)
xTest = pca.transform(xTest)

explain_variance = pca.explained_variance_ratio_

classifier = LogisticRegression()
classifier.fit(xTrain, yTrain)

yPredTrain = classifier.predict(xTrain)
yPredTest = classifier.predict(xTest)

def plot_decision_regions(x, y, classifier, title):
    x1, x2 = np.meshgrid(
        np.arange(start=x[:, 0].min() - 1, stop=x[:, 0].max() + 1, step=0.01),
        np.arange(start=x[:, 1].min() - 1, stop=x[:, 1].max() + 1, step=0.01),
    )
    plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
                alpha=0.75,
                cmap=ListedColormap(('yellow', 'white', 'aquamarine'))
                )
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    for i, j in enumerate(np.unique(y)):
        plt.scatter(x[y == j, 0], x[y == j, 1], c=['red', 'green', 'blue'][i], label=j)
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.show()

plot_decision_regions(xTrain, yTrain, classifier, "Logistic Regression (Training set)")
plot_decision_regions(xTest, yTest, classifier, "Logistic Regression (Test set)")

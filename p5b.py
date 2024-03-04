import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.3, random_state=42, stratify=y)

sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

kn=KNeighborsClassifier(n_neighbors=5, p=2, weights='uniform', algorithm='auto')
kn.fit(X_train_std, y_train)

print(f"Training accuracy score: {kn.score(X_train_std, y_train)}")
print(f"Test accuracy score: {kn.score(X_test_std, y_test)}")

pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier())
param_grid = [{
    'kneighborsclassifier__n_neighbors': [2,3,4,5,6,7,8,9,10],
    'kneighborsclassifier__p': [1,2],
    'kneighborsclassifier__weights': ['uniform', 'distance'],
    'kneighborsclassifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}]

gs = GridSearchCV(pipeline, param_grid=param_grid, scoring='accuracy', refit=True, cv=10, verbose=1, n_jobs=2)
gs.fit(X_train, y_train)
print(f"Best Score: {gs.best_score_}")
print(f"Best Performance: {gs.best_params_}")

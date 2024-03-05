import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

customer_data = pd.read_csv("p7a.csv")
print(f"Shape {customer_data.shape}")
print(f"Head\n{customer_data.head()}")
X = customer_data.iloc[:, 3:4].values
plt.title("Customer Dendograms")
dend = sch.dendrogram(sch.linkage(X, method="ward"))
plt.xlabel("Gender")
plt.ylabel("Euclidean Distances")
plt.show()

cluster = AgglomerativeClustering(n_clusters=5, linkage="ward")
y_hc = cluster.fit_predict(X)
print("Prediction Value:", y_hc)

plt.figure()
plt.scatter(X[:, 0], X[:, -1], c=cluster.labels_, cmap="rainbow")
plt.show()
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Mall_Customers.csv")

# print(df.head())

X = df[["Age","Annual Income (k$)","Spending Score (1-100)"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 5

Kmeans = KMeans(n_clusters=k,random_state=42,n_init=10)
Kmeans.fit(X_scaled)

labels = Kmeans.labels_
df["Cluster"] = labels

print(df.head())

print(Kmeans.cluster_centers_)
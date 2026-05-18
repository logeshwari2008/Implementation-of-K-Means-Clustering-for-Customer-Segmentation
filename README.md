# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the customer dataset and select required features.
2. Choose the number of clusters K and initialize the K-Means model.
3. Fit the model to the data and assign customers to clusters.
4. Display the clustered customer segments and centroids.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: LOGESHWARI N
RegisterNumber:  212225040206
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv("C:/Users/acer/Downloads/Mall_Customers.csv")
print(data.head())
X = data.iloc[:, [3, 4]].values
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)
plt.figure(figsize=(8,6))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], 
kmeans.cluster_centers_[:,1], 
s=300, c='yellow', label='Centroids')
plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

```

## Output:

<img width="690" height="125" alt="image" src="https://github.com/user-attachments/assets/dc628694-eecb-48db-96ad-8b7d020186cd" />

<img width="722" height="468" alt="download" src="https://github.com/user-attachments/assets/aa0bbcbb-2bda-4d2c-b554-5fa3415731c2" />

<img width="695" height="545" alt="download" src="https://github.com/user-attachments/assets/ffba7dc8-1370-4f8d-b810-d01422bad95a" />

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.

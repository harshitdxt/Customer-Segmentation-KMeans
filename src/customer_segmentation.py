import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("data/Mall_Customers.csv")

print("\nFirst Five Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nColumns")
print(df.columns)

print("\nMissing Values")
print(df.isnull().sum())

# Remove duplicate rows if any
df = df.drop_duplicates()

print("\nSummary Statistics")
print(df.describe())


# Age distribution
plt.figure(figsize=(7, 5))
sns.histplot(df["Age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.show()


# Annual income distribution
plt.figure(figsize=(7, 5))
sns.histplot(df["Annual Income (k$)"], bins=20)
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Number of Customers")
plt.show()



# Spending score distribution
plt.figure(figsize=(7, 5))
sns.histplot(df["Spending Score (1-100)"], bins=20)
plt.title("Spending Score Distribution")
plt.xlabel("Spending Score")
plt.ylabel("Number of Customers")
plt.show()



# Gender distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Gender")
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")
plt.show()


from sklearn.cluster import KMeans

X = df.drop("CustomerID", axis=1)
X["Gender"] = X["Gender"].map({"Male": 0, "Female": 1})
print("\nFeatures")
print(X.head())


plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)"
)
plt.title("Annual Income vs Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

from sklearn.cluster import KMeans

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
print("\nFeatures Used")
print(X.head())


wcss = []
for i in range(1, 11):

    model = KMeans(n_clusters=i, random_state=42)

    model.fit(X)

    wcss.append(model.inertia_)

print("\nWCSS Values")
print(wcss)



plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.show()


# Train the K-Means model

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X)


# Add cluster labels to the dataframe
df["Cluster"] = clusters
print("\nFirst Five Rows")
print(df.head())
print("\nCustomers in Each Cluster")
print(df["Cluster"].value_counts().sort_index())




plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="Set1",
    s=80
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    color="black",
    marker="X",
    s=250,
    label="Centroids"
)

plt.title("Customer Segments using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()


print("\nCustomers in Each Cluster")
print(df["Cluster"].value_counts().sort_index())


df.to_csv("output/customer_segments.csv", index=False)
print("\nClustered dataset saved successfully.")
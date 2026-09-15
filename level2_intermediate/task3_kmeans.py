"""
Codveda ML Internship - Level 2, Task 3
K-Means Clustering
Dataset: BigML Telecom Churn (used here WITHOUT the Churn label, since
clustering is unsupervised - we group customers by behavior alone)

Objectives covered:
  1. Load a dataset and preprocess it (scaling)
  2. Apply K-Means and find the optimal number of clusters (elbow method)
  3. Visualize clusters using 2D scatter plots
  4. Interpret the clustering results
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -----------------------------------------------------------------------
# 1. Load data and preprocess
# -----------------------------------------------------------------------
# Combine both files - for customer segmentation we want as many
# customers as possible; the original 80/20 split was only meant for
# supervised train/test, which doesn't apply to unsupervised clustering.
df = pd.concat([
    pd.read_csv("..\\data\\churn-bigml-80.csv"),
    pd.read_csv("..\\data\\churn-bigml-20.csv"),
], ignore_index=True)
print(f"Total customers: {df.shape[0]}")

# Drop identifiers/target: State and Area code aren't behavioral, and
# Churn is deliberately excluded - K-Means must group customers using
# only their usage patterns, with no knowledge of who actually churned.
df["International plan"] = (df["International plan"] == "Yes").astype(int)
df["Voice mail plan"] = (df["Voice mail plan"] == "Yes").astype(int)
churn_labels = df["Churn"].astype(int)  # kept aside only for interpretation later
X = df.drop(columns=["State", "Area code", "Churn"])

# K-Means uses Euclidean distance, so features must be scaled - otherwise
# 'Total day minutes' (range ~0-350) would dominate 'International plan'
# (range 0-1) in the distance calculation.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------------------------------
# 2. Elbow method - find the optimal number of clusters
# -----------------------------------------------------------------------
# For each K, we measure "inertia": the total squared distance from each
# point to its assigned cluster center. Inertia always drops as K rises,
# but the RATE of improvement slows down - the "elbow" in the curve marks
# the point where adding more clusters stops paying off much.
inertias = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(list(k_range), inertias, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia (within-cluster sum of squares)")
plt.title("Elbow Method for Optimal K")
plt.xticks(list(k_range))
plt.tight_layout()
plt.savefig("../results/level2_intermediate/task3_elbow_plot.png", dpi=150)
print("Saved elbow plot to task3_elbow_plot.png")

for k, inertia in zip(k_range, inertias):
    print(f"  K={k}: inertia={inertia:.1f}")

# -----------------------------------------------------------------------
# 3. Fit final K-Means model
# -----------------------------------------------------------------------
# The elbow plot's bend is at K=4 - inertia keeps dropping after that,
# but far more gradually per additional cluster.
CHOSEN_K = 4
kmeans = KMeans(n_clusters=CHOSEN_K, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
df["cluster"] = cluster_labels

print(f"\nUsing K={CHOSEN_K}")
print("Cluster sizes:")
print(df["cluster"].value_counts().sort_index())

# -----------------------------------------------------------------------
# 4. Visualize clusters in 2D
# -----------------------------------------------------------------------
# We have too many features to plot directly, so PCA compresses them
# into the 2 directions that capture the most variance, purely for
# visualization - the clustering itself was done on the full feature set.
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_
print(f"\nPCA: 2 components explain {explained.sum():.1%} of total variance")

plt.figure(figsize=(8, 6))
scatter_df = pd.DataFrame({
    "PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "cluster": cluster_labels
})
sns.scatterplot(data=scatter_df, x="PC1", y="PC2", hue="cluster", palette="Set2", s=25)
plt.title(f"Customer Segments (K-Means, K={CHOSEN_K}) - PCA-reduced view")
plt.xlabel(f"PC1 ({explained[0]:.1%} variance)")
plt.ylabel(f"PC2 ({explained[1]:.1%} variance)")
plt.tight_layout()
plt.savefig("../results/level2_intermediate/task3_cluster_scatter.png", dpi=150)
print("Saved cluster scatter plot to task3_cluster_scatter.png")

# -----------------------------------------------------------------------
# 5. Interpret the clusters
# -----------------------------------------------------------------------
# Profile each cluster by its average feature values, and cross-check
# against the real churn rate (held out of the clustering itself) to see
# whether any cluster naturally lines up with high churn risk.
df["Churn"] = churn_labels
profile_cols = [
    "Total day minutes", "Total eve minutes", "Total night minutes",
    "Customer service calls", "International plan", "Voice mail plan", "Churn"
]
cluster_profile = df.groupby("cluster")[profile_cols].mean().round(2)
print("\nCluster profiles (average values per cluster):")
print(cluster_profile)

df.to_csv("task3_clustered_customers.csv", index=False)
print("\nSaved full clustered dataset to task3_clustered_customers.csv")

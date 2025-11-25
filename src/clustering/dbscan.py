import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from load_clustering_data import load_data_for_clustering

# ---------------------------------------------------------
# 1. Load and Sample Data (Targeting 50k)
# ---------------------------------------------------------
print("📂 Loading Data...")
result = load_data_for_clustering()

# Handle tuple return if necessary
if isinstance(result, tuple):
    df_raw = result[0]
else:
    df_raw = result

# Select only numeric columns
X_raw = df_raw.select_dtypes(include=["number"])

# Sample 50,000 points (or less if dataset is smaller)
sample_size = 50000
if len(X_raw) > sample_size:
    X_sample = X_raw.sample(n=sample_size, random_state=42)
else:
    X_sample = X_raw

print(f"✅ Data sampled. Shape: {X_sample.shape}")

# ---------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------
# Standard Scaling is crucial for DBSCAN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

# ---------------------------------------------------------
# 3. Define Hyperparameter Space for Random Search
# ---------------------------------------------------------
# DBSCAN is not compatible with standard GridSearchCV/RandomizedSearchCV
# because it is transductive (cannot predict on hold-out folds).
# We use ParameterSampler to generate random combinations manually.

param_grid = {
    "eps": np.arange(0.1, 2.0, 0.1),  # Search range for radius
    "min_samples": range(5, 100, 5),  # Search range for density
}

# Define how many random iterations to try
n_iter_search = 20
random_params = list(
    ParameterSampler(param_grid, n_iter=n_iter_search, random_state=42)
)

print(f"🔍 Starting Random Search with {n_iter_search} combinations...")

# ---------------------------------------------------------
# 4. Execution and Evaluation Loop
# ---------------------------------------------------------
results = []

for params in random_params:
    eps = params["eps"]
    ms = params["min_samples"]

    # Run DBSCAN
    db = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1)
    labels = db.fit_predict(X_scaled)

    # Metrics calculation
    # We perform checks to avoid errors if DBSCAN finds only noise (-1) or only 1 cluster
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)

    # Create result dictionary
    res = {
        "eps": eps,
        "min_samples": ms,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "silhouette": -1,  # Default bad scores
        "calinski": -1,
        "davies": 999,
    }

    # Calculate metrics only if clustering is valid (>1 cluster and <100% noise)
    if 1 < n_clusters < len(X_scaled):
        # NOTE: Metrics usually exclude noise points for fair comparison
        mask = labels != -1
        if np.sum(mask) > 0:
            X_core = X_scaled[mask]
            labels_core = labels[mask]

            # Silhouette: Higher is better (-1 to 1)
            res["silhouette"] = silhouette_score(X_core, labels_core)

            # Calinski-Harabasz: Higher is better
            res["calinski"] = calinski_harabasz_score(X_core, labels_core)

            # Davies-Bouldin: Lower is better (0 is best)
            res["davies"] = davies_bouldin_score(X_core, labels_core)

    results.append(res)
    print(
        f"👉 eps={eps:.2f}, min_samples={ms} -> {n_clusters} clusters, Sil: {res['silhouette']:.3f}"
    )

# ---------------------------------------------------------
# 5. Analyze Results
# ---------------------------------------------------------
df_results = pd.DataFrame(results)

# Sort by Silhouette Score (Descending) to find "best"
best_run = df_results.sort_values(by="silhouette", ascending=False).iloc[0]

print("\n🏆 BEST PARAMETERS FOUND:")
print(best_run)

# ---------------------------------------------------------
# 6. Final Visualization (Using Best Params)
# ---------------------------------------------------------
print("\n🎨 Visualizing Best Model...")

# Re-run best model
db_best = DBSCAN(
    eps=best_run["eps"], min_samples=int(best_run["min_samples"]), n_jobs=-1
)
labels_best = db_best.fit_predict(X_scaled)

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
unique_labels = set(labels_best)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]
        label = "Noise"
        marker = "x"
        size = 4
    else:
        label = f"Cluster {k}"
        marker = "o"
        size = 10

    class_member_mask = labels_best == k

    xy = X_pca[class_member_mask]
    plt.scatter(
        xy[:, 0],
        xy[:, 1],
        s=size,
        c=[col],
        marker=marker,
        label=label if k != -1 else None,
    )

plt.title(
    f"DBSCAN: eps={best_run['eps']:.2f}, min_samples={int(best_run['min_samples'])}"
)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()

# Optional: Display top 5 results
print("\nTop 5 Configurations:")
print(df_results.sort_values(by="silhouette", ascending=False).head(5))

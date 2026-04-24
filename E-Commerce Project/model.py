"""
Customer Segmentation using clustering algorithms.
Implements KMeans, DBSCAN, and Hierarchical Clustering.
Includes model training, evaluation, persistence, and cold-start handling.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA

from config import (
    KMEANS_MODEL_PATH,
    DBSCAN_MODEL_PATH,
    HIERARCHICAL_MODEL_PATH,
    SCALER_PATH,
    N_CLUSTERS,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    RANDOM_STATE
)


def load_or_prepare_features(rfm_df=None, data_path="data/rfm_features.csv"):
    """
    Load RFM features from file or DataFrame and prepare for clustering.
    
    Args:
        rfm_df (pd.DataFrame, optional): Pre-loaded RFM DataFrame.
        data_path (str): Path to RFM CSV file.
    
    Returns:
        tuple: (feature_array, feature_df, scaler, feature_names)
    """
    if rfm_df is None:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"RFM features not found at {data_path}. Run data generation first.")
        rfm_df = pd.read_csv(data_path)
    
    # Select numeric features for clustering (exclude customer_id)
    feature_cols = [col for col in rfm_df.columns if col != "customer_id"]
    features = rfm_df[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, rfm_df, scaler, feature_cols


def train_kmeans(features_scaled, n_clusters=N_CLUSTERS):
    """
    Train KMeans clustering model.
    
    Args:
        features_scaled (np.ndarray): Standardized feature array.
        n_clusters (int): Number of clusters.
    
    Returns:
        KMeans: Trained KMeans model.
    """
    print(f"[MODEL] Training KMeans with {n_clusters} clusters...")
    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    model.fit(features_scaled)
    labels = model.labels_
    
    score = silhouette_score(features_scaled, labels)
    print(f"[MODEL] KMeans Silhouette Score: {score:.4f}")
    
    return model


def train_dbscan(features_scaled, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES):
    """
    Train DBSCAN clustering model.
    
    Args:
        features_scaled (np.ndarray): Standardized feature array.
        eps (float): Maximum distance between samples.
        min_samples (int): Minimum samples in neighborhood.
    
    Returns:
        DBSCAN: Trained DBSCAN model.
    """
    print(f"[MODEL] Training DBSCAN (eps={eps}, min_samples={min_samples})...")
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(features_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"[MODEL] DBSCAN found {n_clusters} clusters, {n_noise} noise points.")
    
    if n_clusters > 1:
        score = silhouette_score(features_scaled[labels != -1], labels[labels != -1])
        print(f"[MODEL] DBSCAN Silhouette Score (excl. noise): {score:.4f}")
    
    return model


def train_hierarchical(features_scaled, n_clusters=N_CLUSTERS):
    """
    Train Hierarchical (Agglomerative) Clustering model.
    
    Args:
        features_scaled (np.ndarray): Standardized feature array.
        n_clusters (int): Number of clusters.
    
    Returns:
        AgglomerativeClustering: Trained hierarchical model.
    """
    print(f"[MODEL] Training Hierarchical Clustering with {n_clusters} clusters...")
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    model.fit(features_scaled)
    labels = model.labels_
    
    score = silhouette_score(features_scaled, labels)
    print(f"[MODEL] Hierarchical Silhouette Score: {score:.4f}")
    
    return model


def evaluate_clustering(features_scaled, labels, model_name="Model"):
    """
    Evaluate clustering performance with multiple metrics.
    
    Args:
        features_scaled (np.ndarray): Standardized features.
        labels (np.ndarray): Cluster labels.
        model_name (str): Name of the model for logging.
    
    Returns:
        dict: Evaluation metrics.
    """
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.discard(-1)
    
    if len(unique_labels) < 2:
        return {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}
    
    mask = labels != -1
    features_eval = features_scaled[mask] if mask.sum() < len(labels) else features_scaled
    labels_eval = labels[mask] if mask.sum() < len(labels) else labels
    
    metrics = {
        "silhouette": silhouette_score(features_eval, labels_eval),
        "calinski_harabasz": calinski_harabasz_score(features_eval, labels_eval),
        "davies_bouldin": davies_bouldin_score(features_eval, labels_eval)
    }
    
    print(f"[MODEL] {model_name} Metrics: {metrics}")
    return metrics


def save_models(kmeans_model, dbscan_model, hierarchical_model, scaler):
    """Save all trained models and scaler to disk."""
    joblib.dump(kmeans_model, KMEANS_MODEL_PATH)
    joblib.dump(dbscan_model, DBSCAN_MODEL_PATH)
    joblib.dump(hierarchical_model, HIERARCHICAL_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("[MODEL] All models saved successfully.")


def load_models():
    """Load all trained models and scaler from disk."""
    if not all(os.path.exists(p) for p in [KMEANS_MODEL_PATH, DBSCAN_MODEL_PATH, HIERARCHICAL_MODEL_PATH, SCALER_PATH]):
        raise FileNotFoundError("Model files not found. Please train models first.")
    
    kmeans = joblib.load(KMEANS_MODEL_PATH)
    dbscan = joblib.load(DBSCAN_MODEL_PATH)
    hierarchical = joblib.load(HIERARCHICAL_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    print("[MODEL] All models loaded successfully.")
    return kmeans, dbscan, hierarchical, scaler


def get_cluster_insights(rfm_df, labels, feature_cols):
    """
    Generate insights for each cluster.
    
    Args:
        rfm_df (pd.DataFrame): RFM features DataFrame.
        labels (np.ndarray): Cluster labels.
        feature_cols (list): List of feature column names.
    
    Returns:
        pd.DataFrame: Cluster summary statistics.
    """
    df = rfm_df.copy()
    df["cluster"] = labels
    
    insights = df.groupby("cluster")[feature_cols].mean().round(2)
    insights["count"] = df.groupby("cluster").size()
    insights["percentage"] = (insights["count"] / len(df) * 100).round(1)
    
    return insights


def assign_cluster_new_customer(new_customer_features, model="kmeans"):
    """
    Assign a cluster to a new customer (cold-start handling).
    Uses the trained KMeans model by default.
    
    Args:
        new_customer_features (np.ndarray): Feature vector for new customer.
        model (str): Which model to use ('kmeans', 'dbscan', 'hierarchical').
    
    Returns:
        int: Assigned cluster label.
    """
    kmeans, _, hierarchical, scaler = load_models()
    
    # Standardize features
    features_scaled = scaler.transform(np.array(new_customer_features).reshape(1, -1))
    
    if model == "kmeans":
        return int(kmeans.predict(features_scaled)[0])
    elif model == "hierarchical":
        return int(hierarchical.fit_predict(features_scaled)[0])
    else:
        # For DBSCAN cold-start, fallback to KMeans (DBSCAN doesn't predict)
        return int(kmeans.predict(features_scaled)[0])


def get_cluster_recommendations(cluster_id, rfm_df, labels, top_n=5):
    """
    Get product category recommendations based on cluster preferences.
    
    Args:
        cluster_id (int): Target cluster ID.
        rfm_df (pd.DataFrame): RFM features.
        labels (np.ndarray): Cluster labels.
        top_n (int): Number of top categories to return.
    
    Returns:
        list: Top recommended categories for the cluster.
    """
    df = rfm_df.copy()
    df["cluster"] = labels
    
    cluster_data = df[df["cluster"] == cluster_id]
    cat_cols = [col for col in df.columns if col.startswith("cat_pref_")]
    
    avg_prefs = cluster_data[cat_cols].mean().sort_values(ascending=False)
    top_cats = [col.replace("cat_pref_", "") for col in avg_prefs.head(top_n).index]
    
    return top_cats


def train_and_save_models(rfm_df=None):
    """
    Full pipeline: prepare features, train all models, evaluate, and save.
    
    Args:
        rfm_df (pd.DataFrame, optional): Pre-loaded RFM DataFrame.
    """
    print("=" * 60)
    print("TRAINING CLUSTERING MODELS")
    print("=" * 60)
    
    features_scaled, rfm_df, scaler, feature_cols = load_or_prepare_features(rfm_df)
    
    # Train KMeans
    kmeans = train_kmeans(features_scaled)
    kmeans_metrics = evaluate_clustering(features_scaled, kmeans.labels_, "KMeans")
    
    # Train DBSCAN
    dbscan = train_dbscan(features_scaled)
    dbscan_metrics = evaluate_clustering(features_scaled, dbscan.labels_, "DBSCAN")
    
    # Train Hierarchical
    hierarchical = train_hierarchical(features_scaled)
    hierarchical_metrics = evaluate_clustering(features_scaled, hierarchical.labels_, "Hierarchical")
    
    # Save models
    save_models(kmeans, dbscan, hierarchical, scaler)
    
    # Generate insights for KMeans (primary model)
    insights = get_cluster_insights(rfm_df, kmeans.labels_, feature_cols)
    print("\n[MODEL] KMeans Cluster Insights:")
    print(insights)
    
    # Save insights
    insights_path = "models/cluster_insights.csv"
    insights.to_csv(insights_path)
    print(f"[MODEL] Insights saved to {insights_path}")
    
    print("=" * 60)
    print("MODEL TRAINING COMPLETE")
    print("=" * 60)
    
    return {
        "kmeans": {"model": kmeans, "metrics": kmeans_metrics},
        "dbscan": {"model": dbscan, "metrics": dbscan_metrics},
        "hierarchical": {"model": hierarchical, "metrics": hierarchical_metrics},
        "insights": insights
    }


def get_pca_for_visualization(features_scaled, n_components=2):
    """
    Reduce dimensions using PCA for 2D/3D visualization.
    
    Args:
        features_scaled (np.ndarray): Standardized features.
        n_components (int): Number of components (2 or 3).
    
    Returns:
        np.ndarray: PCA-transformed features.
    """
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    return pca.fit_transform(features_scaled)


if __name__ == "__main__":
    train_and_save_models()


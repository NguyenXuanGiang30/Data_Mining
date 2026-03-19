"""
clustering.py - KMeans / Hierarchical Clustering + Profiling

Module này chịu trách nhiệm:
- Chạy KMeans và Hierarchical clustering
- Xác định số cụm tối ưu (Elbow + Silhouette)
- Mô tả đặc điểm từng cụm
- Rút insight nhóm nguy cơ cao
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from sklearn.preprocessing import StandardScaler


def find_optimal_k(X, k_range=range(2, 11), random_state=42):
    """
    Tìm số cụm tối ưu bằng Elbow + Silhouette.
    
    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Features đã scale.
    k_range : range
    random_state : int
    
    Returns
    -------
    dict
        {"k_range": list, "inertias": list, "silhouettes": list, "best_k": int}
    """
    print("=" * 60)
    print("📊 FINDING OPTIMAL K")
    print("=" * 60)
    
    inertias = []
    silhouettes = []
    dbis = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        sil = silhouette_score(X, labels)
        dbi = davies_bouldin_score(X, labels)
        silhouettes.append(sil)
        dbis.append(dbi)
        print(f"   k={k}: Inertia={kmeans.inertia_:.1f}, Silhouette={sil:.4f}, DBI={dbi:.4f}")
    
    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"\n✅ Best k = {best_k} (Silhouette = {max(silhouettes):.4f}, DBI = {dbis[np.argmax(silhouettes)]:.4f})")
    
    return {
        "k_range": list(k_range),
        "inertias": inertias,
        "silhouettes": silhouettes,
        "dbis": dbis,
        "best_k": best_k,
    }


def run_kmeans(X, n_clusters=3, random_state=42):
    """
    Chạy KMeans clustering.
    
    Parameters
    ----------
    X : pd.DataFrame
    n_clusters : int
    random_state : int
    
    Returns
    -------
    labels : np.ndarray
    kmeans : KMeans
    sil_score : float
    """
    print(f"\n🔵 Running KMeans (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    sil_score = silhouette_score(X, labels)
    dbi_score = davies_bouldin_score(X, labels)
    
    print(f"   Silhouette Score: {sil_score:.4f}")
    print(f"   Davies-Bouldin Index: {dbi_score:.4f}")
    for i in range(n_clusters):
        count = (labels == i).sum()
        print(f"   Cluster {i}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    return labels, kmeans, sil_score, dbi_score


def run_hierarchical(X, n_clusters=3, linkage="ward"):
    """
    Chạy Hierarchical (Agglomerative) Clustering.
    
    Parameters
    ----------
    X : pd.DataFrame
    n_clusters : int
    linkage : str
    
    Returns
    -------
    labels : np.ndarray
    model : AgglomerativeClustering
    sil_score : float
    """
    print(f"\n🟢 Running Hierarchical Clustering (k={n_clusters}, linkage={linkage})...")
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    sil_score = silhouette_score(X, labels)
    dbi_score = davies_bouldin_score(X, labels)
    
    print(f"   Silhouette Score: {sil_score:.4f}")
    print(f"   Davies-Bouldin Index: {dbi_score:.4f}")
    for i in range(n_clusters):
        count = (labels == i).sum()
        print(f"   Cluster {i}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    return labels, model, sil_score, dbi_score


def profile_clusters(df, cluster_labels, target_col="target", feature_cols=None):
    """
    Mô tả đặc điểm từng cụm.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame gốc (chưa scale) để interpret.
    cluster_labels : np.ndarray
    target_col : str
    feature_cols : list
    
    Returns
    -------
    pd.DataFrame
        Bảng profile với mean/std từng cụm.
    """
    df_profile = df.copy()
    df_profile["cluster"] = cluster_labels
    
    if feature_cols is None:
        feature_cols = df.select_dtypes(include="number").columns.tolist()
        feature_cols = [c for c in feature_cols if c not in ["id"]]
    
    # Mean theo cluster
    profile = df_profile.groupby("cluster")[feature_cols].agg(["mean", "std"]).round(2)
    
    # Tỷ lệ bệnh tim theo cluster
    if target_col in df.columns:
        disease_rate = df_profile.groupby("cluster")[target_col].mean().round(4)
        cluster_size = df_profile.groupby("cluster").size()
        
        print("\n📋 CLUSTER PROFILES:")
        print("-" * 50)
        for c in sorted(df_profile["cluster"].unique()):
            print(f"\n  Cluster {c}: {cluster_size[c]} patients "
                  f"({cluster_size[c]/len(df)*100:.1f}%)")
            print(f"  → Tỷ lệ bệnh tim: {disease_rate[c]*100:.1f}%")
            if disease_rate[c] > 0.6:
                print(f"  ⚠️  NHÓM NGUY CƠ CAO!")
            elif disease_rate[c] < 0.3:
                print(f"  ✅ Nhóm nguy cơ thấp")
    
    return profile


def get_cluster_insights(df, cluster_labels, target_col="target"):
    """
    Rút insight từ phân tích cụm.
    
    Returns
    -------
    list of str
        Danh sách insights.
    """
    df_c = df.copy()
    df_c["cluster"] = cluster_labels
    
    insights = []
    
    for c in sorted(df_c["cluster"].unique()):
        cluster_data = df_c[df_c["cluster"] == c]
        size = len(cluster_data)
        
        if target_col in df.columns:
            disease_rate = cluster_data[target_col].mean()
            
            # Tìm đặc trưng nổi bật của cluster
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            numeric_cols = [c2 for c2 in numeric_cols if c2 not in ["id", "target", "num", "cluster"]]
            
            notable_features = []
            for col in numeric_cols:
                cluster_mean = cluster_data[col].mean()
                overall_mean = df[col].mean()
                overall_std = df[col].std()
                
                if overall_std > 0:
                    z = (cluster_mean - overall_mean) / overall_std
                    if abs(z) > 0.5:
                        direction = "cao" if z > 0 else "thấp"
                        notable_features.append(f"{col} {direction}")
            
            insight = (
                f"Cụm {c} ({size} bệnh nhân, {disease_rate*100:.0f}% bệnh tim): "
                f"Đặc trưng bởi {', '.join(notable_features[:5])}"
            )
            insights.append(insight)
    
    return insights

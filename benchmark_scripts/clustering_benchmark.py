import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import gc
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score
)
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import squidpy as sq
import os
import warnings
import scipy.sparse

# Suppress warnings to keep the output clean
warnings.filterwarnings("ignore")


def set_seed(seed=2025):
    """Set a random seed for reproducibility."""
    np.random.seed(seed)
    sc.settings.verbosity = 0
    sc.settings.set_figure_params(dpi=80, facecolor='white')

def _compute_CHAOS(clusterlabel, location):
    """Compute the CHAOS metric for spatial disorganization."""
    clusterlabel = np.array(clusterlabel)
    location = StandardScaler().fit_transform(location)
    
    unique_clusters = np.unique(clusterlabel)
    if len(unique_clusters) < 2:
        return np.nan
        
    total_min_dist = 0.0
    
    for k in unique_clusters:
        cluster_points = location[clusterlabel == k]
        if len(cluster_points) < 2:
            continue
            
        dist_matrix = squareform(pdist(cluster_points))
        np.fill_diagonal(dist_matrix, np.inf)
        total_min_dist += np.sum(np.min(dist_matrix, axis=1))
    
    return total_min_dist / len(clusterlabel)

def _compute_PAS(clusterlabel, location, k=10):
    """Compute the PAS metric for spatial continuity."""
    clusterlabel = np.array(clusterlabel)
    location = StandardScaler().fit_transform(location)
    
    n_samples = location.shape[0]
    if n_samples <= k:
        return np.nan
        
    dist_matrix = squareform(pdist(location))
    pas_count = 0
    
    for i in range(n_samples):
        k_nearest_indices = np.argsort(dist_matrix[i])[1:k+1]
        k_nearest_labels = clusterlabel[k_nearest_indices]
        if np.sum(k_nearest_labels != clusterlabel[i]) > (k / 2):
            pas_count += 1
            
    return pas_count / n_samples

def compute_spatial_metrics(adata, cluster_key):
    """Compute all spatial metrics for a given clustering."""
    if len(adata.obs[cluster_key].unique()) < 2:
        return {'CHAOS': np.nan, 'PAS': np.nan}
    try:
        chaos = _compute_CHAOS(adata.obs[cluster_key], adata.obsm['spatial'])
        pas = _compute_PAS(adata.obs[cluster_key], adata.obsm['spatial'])
        return {'CHAOS': chaos, 'PAS': pas}
    except Exception:
        return {'CHAOS': np.nan, 'PAS': np.nan}

def compute_cluster_metrics(adata, gt_key, pred_key):
    """Compute clustering quality metrics against a ground truth."""
    if len(adata.obs[pred_key].unique()) < 2 or len(adata.obs[gt_key].unique()) < 2:
        return {'ARI': np.nan, 'NMI': np.nan, 'AMI': np.nan, 'Homogeneity': np.nan, 'Num_Clusters': len(adata.obs[pred_key].unique())}
    try:
        return {
            'ARI': adjusted_rand_score(adata.obs[gt_key], adata.obs[pred_key]),
            'NMI': normalized_mutual_info_score(adata.obs[gt_key], adata.obs[pred_key]),
            'AMI': adjusted_mutual_info_score(adata.obs[gt_key], adata.obs[pred_key]),
            'Homogeneity': homogeneity_score(adata.obs[gt_key], adata.obs[pred_key]),
            'Num_Clusters': len(adata.obs[pred_key].unique())
        }
    except Exception:
        return {'ARI': np.nan, 'NMI': np.nan, 'AMI': np.nan, 'Homogeneity': np.nan, 'Num_Clusters': np.nan}

def compute_marker_metrics(adata, cluster_key, top_n=100):
    """Compute spatial autocorrelation for top variable genes."""
    adata = adata.copy()
    try:
        if len(adata.obs[cluster_key].unique()) < 2:
            return {'Moran_I': np.nan, 'Geary_C': np.nan}
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=min(top_n, adata.n_vars - 1), flavor='seurat_v3')
        genes = adata.var[adata.var.highly_variable].index.tolist()
        if not genes:
            return {'Moran_I': np.nan, 'Geary_C': np.nan}
        sq.gr.spatial_neighbors(adata)
        sq.gr.spatial_autocorr(adata, mode="moran", genes=genes, n_jobs=1)
        moran_i = np.nanmedian(adata.uns["moranI"]['I'])
        sq.gr.spatial_autocorr(adata, mode="geary", genes=genes, n_jobs=1)
        geary_c = np.nanmedian(adata.uns["gearyC"]['C'])
        return {'Moran_I': moran_i, 'Geary_C': geary_c}
    except Exception:
        return {'Moran_I': np.nan, 'Geary_C': np.nan}

def process_dataset(adata_path, cluster_files, dataset_name, replicate_id):
    """
    Load a single h5ad dataset and merge it with multiple individual CSV cluster result files.
    Each CSV file represents one clustering method.
    """
    try:
        adata = sc.read(adata_path)
        print(f"    Loading {adata_path} (shape: {adata.shape})")

        # --- Merge multiple cluster CSVs ---
        all_clusters_df = None
        
        for method, file_path in cluster_files.items():
            try:
                # Read the individual CSV for the method
                clusters = pd.read_csv(file_path)
                
                # Handle CSV format (first column is cell barcode)
                if 'Unnamed: 0' in clusters.columns:
                    clusters = clusters.set_index('Unnamed: 0')
                else:
                    print(f"      Warning: 'Unnamed: 0' not in {file_path}, using first column as index.")
                    clusters = clusters.set_index(clusters.columns[0])

                # Rename the cluster column to be specific to the method
                # e.g., 'cluster3_graphst' -> 'graphst'
                original_col_name = clusters.columns[0]
                clusters.rename(columns={original_col_name: method}, inplace=True)

                if all_clusters_df is None:
                    all_clusters_df = clusters
                else:
                    # Merge with the main dataframe
                    all_clusters_df = all_clusters_df.join(clusters, how='outer')

            except Exception as e:
                print(f"      ERROR reading or merging cluster file {file_path}: {e}")
                continue
        
        if all_clusters_df is None or all_clusters_df.empty:
            print("    ERROR: No valid cluster data could be loaded.")
            return None

        print(f"    Successfully merged {len(all_clusters_df.columns)} methods.")
        
        # Find all clustering method columns
        cluster_columns = all_clusters_df.columns.tolist()
        
        # Standardize indices
        adata.obs.index = adata.obs.index.astype(str)
        all_clusters_df.index = all_clusters_df.index.astype(str)
        
        # Find common indices
        common_idx = adata.obs.index.intersection(all_clusters_df.index)
        print(f"    Common indices: {len(common_idx)} out of {len(adata.obs)} cells")
        
        if len(common_idx) == 0:
            print(f"    ERROR: No common indices found!")
            return None
            
        # Subset to common indices
        adata = adata[common_idx].copy()
        clusters = all_clusters_df.loc[common_idx]
        
        # Check for ground truth
        gt_col = None
        if 'ground_truth' in adata.obs:
            gt_col = 'ground_truth'
        elif 'sce.layer_guess' in adata.obs:
            gt_col = 'sce.layer_guess'
        elif 'layer' in adata.obs:
            gt_col = 'layer'
        
        if gt_col is None:
            print(f"    WARNING: No ground truth column found")
            return None
        
        # Use a consistent ground truth column name
        adata.obs['ground_truth'] = adata.obs[gt_col]
        
        # Remove cells with missing ground truth
        adata = adata[~adata.obs['ground_truth'].isna()].copy()
        clusters = clusters.loc[adata.obs.index] # Ensure clusters df matches filtered adata
        
        print(f"    Final data shape: {adata.shape}")
        
        # Return adata with clusters DataFrame for further processing
        return adata, clusters, cluster_columns
    except Exception as e:
        print(f"    ERROR in process_dataset: {e}")
        return None


def find_matching_files(h5ad_dir, run_dir):
    """
    Find the h5ad file matching the run directory and all cluster CSVs within it.
    """
    h5ad_path = None
    cluster_files = {}
    
    # The dataset name is the parent directory of the run_dir
    dataset_name = run_dir.parent.name
    
    # Find the matching h5ad file
    potential_h5ad_path = Path(h5ad_dir) / f"{dataset_name}.h5ad"
    if potential_h5ad_path.exists():
        h5ad_path = str(potential_h5ad_path)
    else:
        print(f"    Warning: No matching h5ad file found for {dataset_name}")
        return None, None

    # Find all cluster result CSV files in the run directory
    for f in run_dir.glob("*.csv"):
        # Extract method from filename, e.g., '..._graphst.csv' -> 'graphst'
        try:
            method_name = f.stem.split('_')[-1]
            if method_name in ['stagate', 'graphst', 'sedr']:
                 cluster_files[method_name] = str(f)
        except IndexError:
            continue
            
    if not cluster_files:
        print(f"    Warning: No cluster CSVs found in {run_dir}")
        return h5ad_path, None

    return h5ad_path, cluster_files


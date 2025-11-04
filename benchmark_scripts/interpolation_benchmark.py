
from pathlib import Path

from scipy.stats import pearsonr, ks_2samp

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
import spateo as st

sc.settings.set_figure_params(dpi=200, facecolor="white")

def to_dense(X):
    """Safely convert a sparse or dense matrix to a dense numpy array."""
    return X.toarray() if hasattr(X, 'toarray') else np.asarray(X)

def safe_mean_calculation(X):
    """Safely calculate mean for both sparse and dense matrices"""
    if sp.issparse(X):
        return np.array(X.mean(axis=0)).flatten()
    else:
        return np.mean(X, axis=0)

def safe_variance_calculation(X):
    """Safely calculate variance for both sparse and dense matrices"""
    if sp.issparse(X):
        # For sparse matrices: Var(X) = E[X²] - E[X]²
        mean_X = np.array(X.mean(axis=0)).flatten()
        mean_X_squared = np.array(X.power(2).mean(axis=0)).flatten()
        return mean_X_squared - mean_X**2
    else:
        return np.var(X, axis=0)

def mean_correlation(real_adata, sim_adata):
    """Calculate correlation between mean gene expressions"""
    try:
        real_mean = safe_mean_calculation(real_adata.X)
        sim_mean = safe_mean_calculation(sim_adata.X)
        
        # Handle edge cases
        if len(real_mean) == 0 or len(sim_mean) == 0:
            return 0.0
        
        # Remove genes with zero variance in both datasets
        mask = (real_mean > 1e-10) & (sim_mean > 1e-10)
        if np.sum(mask) < 10:  # Need at least 10 genes for meaningful correlation
            return 0.0
            
        corr, _ = pearsonr(real_mean[mask], sim_mean[mask])
        return corr if not np.isnan(corr) else 0.0
        
    except Exception as e:
        print(f"Error in mean_correlation: {e}")
        return 0.0

def variance_correlation(real_adata, sim_adata):
    """Calculate correlation between gene variances"""
    try:
        real_var = safe_variance_calculation(real_adata.X)
        sim_var = safe_variance_calculation(sim_adata.X)
        
        # Handle edge cases
        if len(real_var) == 0 or len(sim_var) == 0:
            return 0.0
            
        # Remove genes with zero variance in both datasets
        mask = (real_var > 1e-10) & (sim_var > 1e-10)
        if np.sum(mask) < 10:
            return 0.0
            
        corr, _ = pearsonr(real_var[mask], sim_var[mask])
        return corr if not np.isnan(corr) else 0.0
        
    except Exception as e:
        print(f"Error in variance_correlation: {e}")
        return 0.0

def zero_proportion_ks_test(real_adata, sim_adata):
    """KS test for zero proportion distributions across genes"""
    try:
        if sp.issparse(real_adata.X):
            real_zero = np.array((real_adata.X == 0).sum(axis=0) / real_adata.shape[0]).flatten()
        else:
            real_zero = np.mean(real_adata.X == 0, axis=0)
            
        if sp.issparse(sim_adata.X):
            sim_zero = np.array((sim_adata.X == 0).sum(axis=0) / sim_adata.shape[0]).flatten()
        else:
            sim_zero = np.mean(sim_adata.X == 0, axis=0)
        
        if len(real_zero) == 0 or len(sim_zero) == 0:
            return 1.0
            
        ks, _ = ks_2samp(real_zero, sim_zero)
        return ks if not np.isnan(ks) else 1.0
        
    except Exception as e:
        print(f"Error in zero_proportion_ks_test: {e}")
        return 1.0

def _compute_morans_for_slice(adata, genes, k=5):
    """Helper function to compute Moran's I for all specified genes."""
    coords = adata.obsm.get('spatial')
    if coords is None: raise ValueError("Spatial coordinates not found in .obsm['spatial']")
    X = to_dense(adata[:, genes].X)
    
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(coords))).fit(coords)
    W = nbrs.kneighbors_graph(coords).toarray()
    np.fill_diagonal(W, 0)
    
    W_sum = W.sum()
    if W_sum == 0: return np.full(len(genes), np.nan)
    
    n = W.shape[0]
    e = X - X.mean(axis=0)
    
    num = (n / W_sum) * np.einsum('ij,ik,jk->k', W, e, e)
    den = (e ** 2).sum(axis=0) / n
    den[den == 0] = 1.0
    
    return num / den

def calculate_structural_similarity(ad_ref, ad_pred, genes):
    """Calculates similarity based on spatial autocorrelation (Moran's I)."""
    print('  Calculating Metric 2: Structural Similarity...')
    try:
        morans_ref = _compute_morans_for_slice(ad_ref, genes)
        morans_pred = _compute_morans_for_slice(ad_pred, genes)
        valid = ~np.isnan(morans_ref) & ~np.isnan(morans_pred)
        if valid.sum() < 2: return np.nan
        return float(pearsonr(morans_ref[valid], morans_pred[valid])[0])
    except Exception as e:
        print(f'    ! Moran computation failed: {e}')
        return np.nan

def _compute_gene_phash(coords, expression, grid_size=8):
    """Helper function to compute a perceptual hash for a single gene pattern."""
    coords_norm = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0) + 1e-6)
    grid_coords = (coords_norm * (grid_size - 1)).astype(int)
    
    grid = np.zeros((grid_size, grid_size)); counts = np.zeros_like(grid)
    for (r, c), val in zip(grid_coords, expression):
        grid[r, c] += val; counts[r, c] += 1
    
    counts[counts == 0] = 1
    grid_avg = grid / counts
    return (grid_avg >= np.median(grid_avg)).astype(int).flatten()

def calculate_perceptual_hash_similarity(ad_ref, ad_pred, genes, grid_size=8):
    """Calculates similarity based on perceptual hashing of gene patterns."""
    print('  Calculating Metric 3: Perceptual Hash Similarity...')
    try:
        coords_ref, coords_pred = ad_ref.obsm['spatial'], ad_pred.obsm['spatial']
        X_ref, X_pred = to_dense(ad_ref[:, genes].X), to_dense(ad_pred[:, genes].X)
        
        total_sim = 0.0
        for i in range(len(genes)):
            hash_ref = _compute_gene_phash(coords_ref, X_ref[:, i], grid_size)
            hash_pred = _compute_gene_phash(coords_pred, X_pred[:, i], grid_size)
            total_sim += 1.0 - (np.sum(hash_ref != hash_pred) / (grid_size ** 2))
            
        return float(total_sim / len(genes)) if genes else np.nan
    except Exception as e:
        print(f'    ! Perceptual hash failed: {e}')
        return np.nan


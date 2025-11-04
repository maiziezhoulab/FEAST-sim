import numpy as np
import scipy.sparse as sp
from scipy.stats import pearsonr, ks_2samp, spearmanr
import scanpy as sc
import squidpy as sq
import pandas as pd
import warnings
import gc
import os
import multiprocessing
from contextlib import contextmanager
import signal
import time

warnings.filterwarnings('ignore')

class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds):
    """Context manager for timeout functionality"""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler and a alarm
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old signal handler
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


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

def spot_zero_proportion_ks_test(real_adata, sim_adata):
    """KS test for zero proportion distributions across spots"""
    try:
        if sp.issparse(real_adata.X):
            real_zero = np.array((real_adata.X == 0).sum(axis=1) / real_adata.shape[1]).flatten()
        else:
            real_zero = np.mean(real_adata.X == 0, axis=1)
            
        if sp.issparse(sim_adata.X):
            sim_zero = np.array((sim_adata.X == 0).sum(axis=1) / sim_adata.shape[1]).flatten()
        else:
            sim_zero = np.mean(sim_adata.X == 0, axis=1)
        
        if len(real_zero) == 0 or len(sim_zero) == 0:
            return 1.0
            
        ks, _ = ks_2samp(real_zero, sim_zero)
        return ks if not np.isnan(ks) else 1.0
        
    except Exception as e:
        print(f"Error in spot_zero_proportion_ks_test: {e}")
        return 1.0

def relative_error_mean(real_adata, sim_adata):
    """Calculate mean relative error between gene expressions"""
    try:
        real_mean = safe_mean_calculation(real_adata.X)
        sim_mean = safe_mean_calculation(sim_adata.X)
        
        if len(real_mean) == 0 or len(sim_mean) == 0:
            return 1.0
            
        relative_error = np.abs(real_mean - sim_mean) / (real_mean + 1e-10)
        return np.mean(relative_error) if not np.isnan(np.mean(relative_error)) else 1.0
        
    except Exception as e:
        print(f"Error in relative_error_mean: {e}")
        return 1.0

def spatial_corr_optimized(adata, max_genes=300, n_perms=10, timeout_seconds=300):
    """
    Optimized spatial autocorrelation calculation with timeout and error handling
    """

    with timeout(timeout_seconds):
        adata_work = adata.copy()
        
        total_genes = len(adata_work.var_names)
        print(f"Total genes in dataset: {total_genes}")
        
        # Smart gene selection strategy
        if total_genes <= 500:
            # For small datasets, use all genes
            selected_genes = adata_work.var_names
            print(f"Small dataset: using all {total_genes} genes")
            n_perms_adjusted = min(n_perms, 10)  # Limit permutations for small datasets
        elif total_genes <= max_genes:
            # Use all genes if within limit
            selected_genes = adata_work.var_names
            print(f"Using all {total_genes} genes")
            n_perms_adjusted = n_perms
        else:
            # Select top expressed genes for large datasets
            print(f"Large dataset: selecting top {max_genes} genes")
            gene_means = safe_mean_calculation(adata_work.X)
            top_genes_idx = np.argsort(gene_means)[-max_genes:]
            selected_genes = adata_work.var_names[top_genes_idx]
            n_perms_adjusted = max(5, n_perms // 2)  # Reduce permutations for speed
        
        print(f"Using {n_perms_adjusted} permutations for {len(selected_genes)} genes")
        
        # **CRITICAL FIX**: Use n_jobs=1 to avoid multiprocessing issues
        # The original error was caused by using n_jobs=-1 in a forked process
        
        # Calculate Moran's I
        sq.gr.spatial_autocorr(
            adata_work,
            mode="moran",
            genes=selected_genes,
            n_perms=n_perms_adjusted,
            n_jobs=1,  # **FIXED**: Use single thread to avoid fork issues
        )
        
        # Calculate Geary's C
        sq.gr.spatial_autocorr(
            adata_work,
            mode="geary", 
            genes=selected_genes,
            n_perms=n_perms_adjusted,
            n_jobs=1,  # **FIXED**: Use single thread to avoid fork issues
        )
        
        # Handle missing values
        if "moranI" in adata_work.uns:
            adata_work.uns["moranI"] = adata_work.uns["moranI"].fillna(0)
        else:
            # Create dummy results if calculation failed
            dummy_df = pd.DataFrame({
                'I': [0.0] * len(selected_genes),
                'pval_norm': [1.0] * len(selected_genes)
            }, index=selected_genes)
            adata_work.uns["moranI"] = dummy_df
            
        if "gearyC" in adata_work.uns:
            adata_work.uns["gearyC"] = adata_work.uns["gearyC"].fillna(1)
        else:
            # Create dummy results if calculation failed
            dummy_df = pd.DataFrame({
                'C': [1.0] * len(selected_genes),
                'pval_norm': [1.0] * len(selected_genes)
            }, index=selected_genes)
            adata_work.uns["gearyC"] = dummy_df
        
        return adata_work
        


import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as stats
from scipy.sparse import issparse
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from concurrent.futures import ThreadPoolExecutor
import os
import gc
import glob


def calculate_alignment_accuracy(pi, gt):
    """
    Calculate alignment accuracy, precision, recall, and F1 score
    
    Parameters:
        pi: Alignment probability matrix (n_spots1 × n_spots2)
        gt: Ground truth matrix (n_spots1 × n_spots2)
    
    Returns:
        Dictionary with accuracy, precision, recall, f1_score
    """
    # Generate prediction matrix
    pred = np.zeros_like(pi)
    rows = np.arange(pi.shape[0])
    cols = np.argmax(pi, axis=1)
    pred[rows, cols] = 1
    
    # Calculate confusion matrix elements
    tp = (pred * gt).sum()  # True positives
    fp = (pred * (1 - gt)).sum()  # False positives
    fn = ((1 - pred) * gt).sum()  # False negatives
    
    total_gt = gt.sum()  # Total pairs in ground truth
    
    # Calculate metrics
    accuracy = tp / total_gt if total_gt > 0 else 0.0  # Use GT total as denominator
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def calculate_alignment_consistency(pi):
    """
    Calculate bidirectional alignment consistency
    If spot A → B and B → A, then it's consistent
    """
    # Forward matching: slice1 → slice2
    forward_match = np.argmax(pi, axis=1)
    
    # Backward matching: slice2 → slice1
    backward_match = np.argmax(pi, axis=0)
    
    # Check bidirectional consistency
    consistent = 0
    for i in range(len(forward_match)):
        j = forward_match[i]
        if j < len(backward_match) and backward_match[j] == i:
            consistent += 1
    
    return consistent / len(forward_match) if len(forward_match) > 0 else 0.0

def calculate_region_accuracy(slice1, slice2, pi, region_key='sce.layer_guess'):
    """Calculate region alignment accuracy (corrected version)"""
    # Check if region labels exist
    if region_key not in slice1.obs or region_key not in slice2.obs:
        return np.nan
    
    # Generate ground truth mask: only consider spots that should have matches
    common_ids = np.intersect1d(slice1.obs_names, slice2.obs_names)
    gt_mask = np.isin(slice1.obs_names, common_ids)
    
    # Get predicted mapping
    pred_mapping = np.argmax(pi, axis=1)
    valid_mask = (pred_mapping < slice2.shape[0]) & gt_mask  # Key fix: add gt_mask
    
    # Extract region labels
    slice1_regions = slice1.obs[region_key].values
    slice2_regions = slice2.obs[region_key].values
    
    # Calculate region matching accuracy
    correct = 0
    total = np.sum(valid_mask)  # Use GT-filtered total
    
    if total == 0:
        return 0.0
    
    for i in np.where(valid_mask)[0]:
        if slice1_regions[i] == slice2_regions[pred_mapping[i]]:
            correct += 1
    
    return correct / total

def spatial_ari(slice1, slice2, pi, n_neighbors=15, region_key='sce.layer_guess'):
    """Calculate ARI based on alignment-predicted spot correspondences (corrected version)"""
    # Check if region labels exist
    if region_key not in slice1.obs or region_key not in slice2.obs:
        return np.nan
    
    # Generate ground truth mask
    common_ids = np.intersect1d(slice1.obs_names, slice2.obs_names)
    gt_mask = np.isin(slice1.obs_names, common_ids)
    
    # Get the predicted mapping from alignment matrix
    pred_mapping = np.argmax(pi, axis=1)
    valid_mask = (pred_mapping < slice2.shape[0]) & gt_mask  # Key fix: add gt_mask
    
    # Extract labels for valid mappings
    slice1_labels = slice1.obs[region_key].values[valid_mask]
    slice2_labels = slice2.obs[region_key].values[pred_mapping[valid_mask]]
    
    # Calculate ARI based on the alignment
    if len(slice1_labels) == 0:
        return 0.0
    
    return adjusted_rand_score(slice1_labels, slice2_labels)

def batch_pearson(true_expr, pred_expr, batch_size=1000):
    """Calculate gene expression correlation in batches"""
    correlations = []
    for i in range(0, len(true_expr), batch_size):
        batch_true = true_expr[i:i+batch_size]
        batch_pred = pred_expr[i:i+batch_size]
        
        # Vectorized calculation
        mean_true = np.mean(batch_true, axis=1, keepdims=True)
        mean_pred = np.mean(batch_pred, axis=1, keepdims=True)
        
        numerator = np.sum((batch_true - mean_true)*(batch_pred - mean_pred), axis=1)
        denominator = np.sqrt(np.sum((batch_true - mean_true)**2, axis=1)) * \
                    np.sqrt(np.sum((batch_pred - mean_pred)**2, axis=1))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            batch_corr = np.divide(numerator, denominator)
            batch_corr[np.isnan(batch_corr)] = 0
        
        correlations.extend(batch_corr)
    
    return np.mean(correlations) if correlations else 0.0

def process_alignment_matrix(pi, slice1, slice2):
    """Process alignment matrix and calculate metrics"""
    # Data preprocessing
    pi = np.nan_to_num(pi, nan=0)
    pi = np.clip(pi, 0, None)
    
    # Generate ground truth matrix
    common_ids = np.intersect1d(slice1.obs_names, slice2.obs_names)
    gt = np.zeros((len(slice1.obs_names), len(slice2.obs_names)))
    
    id_to_idx = {id_:i for i, id_ in enumerate(slice2.obs_names)}
    for i, id_ in enumerate(slice1.obs_names):
        if id_ in id_to_idx:
            gt[i, id_to_idx[id_]] = 1
    
    print(f"    Ground truth: {int(gt.sum())} actual pairs out of {len(slice1.obs_names)} spots in slice1")
    print(f"    Overlap rate: {gt.sum() / len(slice1.obs_names) * 100:.1f}%")
    
    # Calculate improved alignment metrics
    alignment_metrics = calculate_alignment_accuracy(pi, gt)
    
    # Calculate other metrics (all with corrected denominators)
    metrics = {
        **alignment_metrics,  # accuracy, precision, recall, f1_score
        'alignment_consistency': calculate_alignment_consistency(pi),
        'ari': spatial_ari(slice1, slice2, pi),
        'region_accuracy': calculate_region_accuracy(slice1, slice2, pi)
    }
    
    # Calculate gene expression correlation in parallel
    with ThreadPoolExecutor() as executor:
        future = executor.submit(calculate_ge_correlation, slice1, slice2, pi)
        metrics['ge_correlation'] = future.result()
    
    return metrics

def calculate_ge_correlation(slice1, slice2, pi):
    """Calculate gene expression correlation (optimized version)"""
    # Generate ground truth mask
    common_ids = np.intersect1d(slice1.obs_names, slice2.obs_names)
    gt_mask = np.isin(slice1.obs_names, common_ids)
    
    # Get mapped expression matrix
    max_indices = np.argmax(pi, axis=1)
    valid_mask = (max_indices < slice2.shape[0]) & gt_mask  # Add gt_mask filter
    
    # Convert to dense matrix
    if issparse(slice1.X):
        mapped_expr = slice1.X[valid_mask].A
    else:
        mapped_expr = slice1.X[valid_mask]
    
    if issparse(slice2.X):
        true_expr = slice2.X[max_indices[valid_mask]].A
    else:
        true_expr = slice2.X[max_indices[valid_mask]]
    
    # Filter zero expression
    non_zero_mask = (np.sum(true_expr, axis=1) > 0) & (np.sum(mapped_expr, axis=1) > 0)
    return batch_pearson(true_expr[non_zero_mask], mapped_expr[non_zero_mask])

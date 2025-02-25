import pandas as pd
import numpy as np
from multiprocessing import Pool
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist, pdist, squareform

def calculate_distance_chunk(args):
    """
    Computes pairwise distances between a chunk of samples and anchor points using the specified metric.

    Parameters:
    - args (tuple): A tuple containing:
        - chunk (np.ndarray): Descriptors of a subset of samples for which distances are computed.
        - anchor_cols (np.ndarray): Descriptors of the anchor anchor points used for distance calculations.
        - distance_metric (str): The distance metric to use ('mhfp_jaccard' or a standard metric supported by `cdist`).

    Returns:
    - np.ndarray: A 2D array of computed distances between the chunk samples and the anchor points.
    """
    chunk, anchor_cols, distance_metric = args

    if distance_metric == 'mhfp_jaccard':
        # Compute distances using mhfp_jaccard
        distances = [
            [mhfp_jaccard(fp1, fp2) for fp2 in anchor_cols] for fp1 in chunk
        ]
    else:
        # Compute distances using cdist for standard metrics
        distances = cdist(chunk, anchor_cols, metric=distance_metric)
    return np.array(distances)

def calculate_distance_parallel(desc, train_idx, anchors_idx, distance, cpus):
    """
    Computes pairwise distances between training samples and anchor points in parallel.

    Parameters:
    - desc (pd.DataFrame): Molecular descriptors of samples.
    - train_idx (list): Indices of training samples.
    - anchors_idx (list): Indices of anchor points.
    - distance (str): The distance metric to use.
    - cpus: Number of CPUs available for parallel computation.

    Returns:
    - pd.DataFrame: A DataFrame containing the computed distances, with training sample indices as rows 
      and anchor indices as columns.
    """
    train_cols = desc.loc[train_idx, :].values
    anchor_cols = desc.loc[anchors_idx, :].values
    # split into subsets for parallel computing
    if cpus > 1: # process with multiple cpus
        train_chunks = np.array_split(train_cols, cpus)
        chunk_args = [(chunk, anchor_cols, distance) for chunk in train_chunks]
        with Pool(processes=cpus) as pool:
            dist_chunks = pool.map(calculate_distance_chunk, chunk_args)
        dist = np.vstack(dist_chunks)
    else:
        dist = calculate_distance_chunk([train_cols, anchor_cols, distance])
    # 
    dist = pd.DataFrame(dist, index=train_idx, columns=anchors_idx)
    return dist


def mhfp_jaccard(fp1, fp2):
    """
    Computes the Jaccard distance for MHFP.

    Parameters:
    - fp1 (np.ndarray): First molecular fingerprint.
    - fp2 (np.ndarray): Second molecular fingerprint.

    Returns:
    - float: Jaccard distance between the two fingerprints.
    """
    return 1.0 - float(np.count_nonzero(fp1 == fp2)) / float(len(fp1))


def calculate_distances(data, distance):
    """
    Computes the pairwise distance matrix based on the specified metric.

    Parameters:
    - data (np.ndarray): Descriptors for which distances will be calculated.
    - args (Namespace): A configuration object containing model parameters.

    Returns:
    - np.ndarray: A square pairwise distance matrix.
    """

    if distance in ['jaccard', 'euclidean', 'manhattan', 'cosine', 'chebyshev']:
        return pairwise_distances(data, metric=distance, n_jobs=-1)
    elif distance == 'mhfp_jaccard':
        return squareform(pdist(data, lambda u, v: mhfp_jaccard(u,v)))
    else:
        raise ValueError("Unsupported metric")

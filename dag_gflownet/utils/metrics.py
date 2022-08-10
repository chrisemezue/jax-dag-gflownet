"""
The code is adapted from:
https://github.com/larslorch/dibs/blob/master/dibs/metrics.py

MIT License

Copyright (c) 2021 Lars Lorch

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
from typing import Optional
from sklearn import metrics


def expected_shd(posterior, ground_truth):
    """Compute the Expected Structural Hamming Distance.

    This function computes the Expected SHD between a posterior approximation
    given as a collection of samples from the posterior, and the ground-truth
    graph used in the original data generation process.

    Parameters
    ----------
    posterior : np.ndarray instance
        Posterior approximation. The array must have size `(B, N, N)`, where `B`
        is the number of sample graphs from the posterior approximation, and `N`
        is the number of variables in the graphs.

    ground_truth : np.ndarray instance
        Adjacency matrix of the ground-truth graph. The array must have size
        `(N, N)`, where `N` is the number of variables in the graph.

    Returns
    -------
    e_shd : float
        The Expected SHD.
    """
    # Compute the pairwise differences
    diff = np.abs(posterior - np.expand_dims(ground_truth, axis=0))
    diff = diff + diff.transpose((0, 2, 1))

    # Ignore double edges
    diff = np.minimum(diff, 1)
    shds = np.sum(diff, axis=(1, 2)) / 2

    return np.mean(shds)


def expected_edges(posterior):
    """Compute the expected number of edges.

    This function computes the expected number of edges in graphs sampled from
    the posterior approximation.

    Parameters
    ----------
    posterior : np.ndarray instance
        Posterior approximation. The array must have size `(B, N, N)`, where `B`
        is the number of sample graphs from the posterior approximation, and `N`
        is the number of variables in the graphs.

    Returns
    -------
    e_edges : float
        The expected number of edges.
    """
    num_edges = np.sum(posterior, axis=(1, 2))
    return np.mean(num_edges)


def threshold_metrics(posterior, ground_truth):
    """Compute threshold metrics (e.g. AUROC, Precision, Recall, etc...).

    Parameters
    ----------
    posterior : np.ndarray instance
        Posterior approximation. The array must have size `(B, N, N)`, where `B`
        is the number of sample graphs from the posterior approximation, and `N`
        is the number of variables in the graphs.

    ground_truth : np.ndarray instance
        Adjacency matrix of the ground-truth graph. The array must have size
        `(N, N)`, where `N` is the number of variables in the graph.

    Returns
    -------
    metrics : dict
        The threshold metrics.
    """
    # Expected marginal edge features
    p_edge = np.mean(posterior, axis=0)
    p_edge_flat = p_edge.reshape(-1)
    
    gt_flat = ground_truth.reshape(-1)

    # Threshold metrics 
    fpr, tpr, _ = metrics.roc_curve(gt_flat, p_edge_flat)
    roc_auc = metrics.auc(fpr, tpr)
    precision, recall, _ = metrics.precision_recall_curve(gt_flat, p_edge_flat)
    prc_auc = metrics.auc(recall, precision)
    ave_prec = metrics.average_precision_score(gt_flat, p_edge_flat)
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'prc_auc': prc_auc,
        'ave_prec': ave_prec,
    }


def compute_ATE(posterior,ground_truth):
    """
    Computes the ATE RMSE between the graph samples in the posterior and the ground truth DAG. 
    """ 

    

def get_ate_from_samples(
    intervened_samples: np.ndarray,
    baseline_samples: np.ndarray,
    normalise: bool = False,
    processed: bool = True,
):
    """
    Computes ATE E[y | do(x)=a] - E[y] from samples of y from p(y | do(x)=a) and p(y)

    Args:
        intervened_samples: np.ndarray of shape (Nsamples, observation_dimension) containing samples from the intervened distribution p(y | do(x)=a)
        baseline_samples: np.ndarray of shape (Nsamples, observation_dimension) containing samples from the non-intervened distribution p(y)
        normalise: boolean indicating whether to normalise samples by their maximum and minimum values
        processed: whether the data has been processed (which affects the column numbering)
    """
    intervened_mean = intervened_samples.mean(axis=0)
    baseline_mean = baseline_samples.mean(axis=0)

    return intervened_mean - baseline_mean    

def calculate_rmse(a: np.ndarray, b: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Calculates the root mean squared error (RMSE) between arrays `a` and `b`.

    Args:
        a (ndarray): Array used for error calculation
        b (ndarray): Array used for error calculation
        axis (int): Axis upon which to calculate mean

    Returns: (ndarray) RMSE value taken along axis `axis`.
    """
    return np.sqrt(np.mean(np.square(np.subtract(a, b)), axis=axis))


 # Calculate RMSE   -> Can also do `atc`
 # BASELINE_FOLDER = '/home/mila/c/chris.emezue/scratch/baselines'
 # decide intervention and target variable and intervention value 
 # for each baseline in baselines:
 # for each seed in baseline
 # read data: 
 # read true graph (and have in networkx DiGraph format.)
 #  read posterior numpy
 # for each posterior:
 #  convert to a form of digraph
 #  put graph inside doWhy
 #  get estimand and calculate estimate
 # get RMSE between pred and real 
 # outcome: CSV where columns are baselines 
 # |baseline|value|seed|

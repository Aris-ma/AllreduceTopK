import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import networkx as nx
from mpmath import mp

def init_data(n=6, d=5, L=200, seed=42, sigma_h=10):
    """
    Initializes heterogeneous data for multiple nodes.

    Args:
        n (int): Number of nodes.
        d (int): Dimension of the data features.
        L (int): Number of data samples per node.
        seed (int): Random seed for reproducibility.
        sigma_h (float): Standard deviation for creating personalized optimal parameters.

    Returns:
        tuple: A tuple containing:
            - h (np.ndarray): Input data for each node. Shape: (n, L, d).
            - y (np.ndarray): Labels for each node. Shape: (n, L).
            - x_opt (np.ndarray): The global optimal parameter. Shape: (1, d).
            - x_star (np.ndarray): Personalized optimal parameters for each node. Shape: (n, d).
    """
    np.random.seed(seed)
    # Generate a global optimal parameter
    x_opt = np.random.normal(size=(1, d))  # Shape: (1, d)
    # Generate personalized optimal parameters by adding noise to the global one
    x_star = x_opt + sigma_h * np.random.normal(size=(n, d))  # Shape: (n, d)
    # Generate input data from a normal distribution
    h = np.random.normal(size=(n, L, d))  # Shape: (n, L, d)
    # Initialize labels
    y = np.zeros((n, L))  # Shape: (n, L)

    # Generate labels based on a logistic regression model
    for i in range(n):
        for l in range(L):
            z = np.random.uniform(0, 1)
            # This is a way to sample from a Bernoulli distribution with parameter p = 1 / (1 + exp(-<h, x_star>))
            if 1 / z > 1 + np.exp(-np.inner(h[i, l, :], x_star[i])):
                y[i, l] = 1
            else:
                y[i, l] = -1
    return (h, y, x_opt, x_star)


def init_x_func(n=6, d=10, seed=42):
    """
    Initializes the parameters for each node.

    Args:
        n (int): Number of nodes.
        d (int): Dimension of the parameters.
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Initialized parameters for each node. Shape: (n, d).
    """
    np.random.seed(seed)
    # Initialize parameters from a scaled normal distribution
    return 0.01 * np.random.normal(size=(n, d))  # Shape: (n, d)


def init_global_data(d=5, L_total=200, seed=42):
    """
    Initializes homogeneous (global) data.

    Args:
        d (int): Dimension of the data features.
        L_total (int): Total number of data samples.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing:
            - h (np.ndarray): Global input data. Shape: (L_total, d).
            - y (np.ndarray): Global labels. Shape: (L_total,).
            - x_opt (np.ndarray): The global optimal parameter. Shape: (1, d).
    """
    np.random.seed(seed)
    # Generate a global optimal parameter
    x_opt = np.random.normal(size=(1, d))  # Shape: (1, d)

    # Generate global input data
    h = np.random.normal(size=(L_total, d))  # Shape: (L_total, d)

    # Initialize global labels
    y = np.zeros(L_total)  # Shape: (L_total,)
    # Generate labels based on a logistic regression model
    for l in range(L_total):
        z = np.random.uniform(0, 1)
        # Sample from a Bernoulli distribution
        if 1 / z > 1 + np.exp(-np.dot(h[l], x_opt.flatten())):
            y[l] = 1
        else:
            y[l] = -1

    return h, y, x_opt


def distribute_data(h, y, n):
    """
    Distributes global data among n nodes.

    Args:
        h (np.ndarray): Global input data. Shape: (L_total, d).
        y (np.ndarray): Global labels. Shape: (L_total,).
        n (int): Number of nodes to distribute data to.

    Returns:
        tuple: A tuple containing:
            - h_tilde (np.ndarray): Reshaped input data for each node. Shape: (n, L_per_node, d).
            - y_tilde (np.ndarray): Reshaped labels for each node. Shape: (n, L_per_node).
    """
    L_total = h.shape[0]
    assert L_total % n == 0, "L_total must be divisible by n"
    L_per_node = L_total // n

    # Reshape the data to be distributed among nodes
    h_tilde = h.reshape(n, L_per_node, -1)  # Shape: (n, L_per_node, d)
    y_tilde = y.reshape(n, L_per_node)      # Shape: (n, L_per_node)

    return h_tilde, y_tilde


def init_hetero_global_data(n=6, d=5, L_per_node=200, seed=42):
    """
    每个节点拥有不同分布的输入数据（feature heterogeneity）
    """
    np.random.seed(seed)
    
    # 每个节点的数据来自不同的 N(mu_i, I)
    mu_list = np.random.normal(size=(n, d))
    
    h = np.zeros((n, int(L_per_node), d))
    y = np.zeros((n, int(L_per_node)))
    
    # 一个共同的全局最优 x_opt
    x_opt = np.random.normal(size=(1, d))
    
    for i in range(n):
        # 节点 i 的输入分布不同
        h[i] = np.random.normal(loc=mu_list[i], scale=1.0, size=(int(L_per_node), d))
        
        # logistic 标签
        for l in range(int(L_per_node)):
            p = 1 / (1 + np.exp(-np.dot(h[i, l], x_opt.flatten())))
            y[i, l] = 1 if np.random.rand() < p else -1
    
    return h, y, x_opt

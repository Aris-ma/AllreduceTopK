import torch
import torch.nn as nn
import networkx as nx
from mpmath import mp
import matplotlib.pyplot as plt
import numpy as np

def get_right_perron(W):
    c = np.linalg.eig(W) 
    eigenvalues = c[0]
    max_eigen = np.abs(eigenvalues).argmax()
    vector = c[1][:,max_eigen]
    return np.abs(vector / np.sum(vector))

def get_left_perron(W):
    return get_right_perron(W.T)

def compute_kappa_row(A):
    pi=get_left_perron(A)
    return np.max(pi)/np.min(pi)

def compute_kappa_col(B):
    pi=get_right_perron(B)
    return np.max(pi)/np.min(pi)

def compute_2st_eig_value(A):
    return abs(np.linalg.eigvals(A)[1])

def compute_beta_row(A, precision=64):
    mp.dps = precision
    n = A.shape[0]
    pi = get_left_perron(A)
    one = np.ones(n)
    if not nx.is_strongly_connected(nx.DiGraph(A)):
        print("not strongly connected")
    matrix = A - np.outer(one, pi)
    diag1 = np.diag(np.sqrt(pi))
    diag1_inverse = np.diag(1 / np.sqrt(pi))
    result = np.linalg.norm(diag1 @ matrix @ diag1_inverse, 2)
    return min(result, 1)

def compute_beta_col(B, precision=64):
    mp.dps = precision
    n = B.shape[0]
    pi = get_right_perron(B)
    one = np.ones(n)
    if not nx.is_strongly_connected(nx.DiGraph(B)):
        print("not strongly connected")
    matrix = B - np.outer(pi, one)
    diag1 = np.diag(np.sqrt(pi))
    diag1_inverse = np.diag(1 / np.sqrt(pi))
    result = np.linalg.norm(diag1_inverse @ matrix @ diag1, 2)
    return min(result, 1)

def compute_S_A_row(A):
    kappa=compute_kappa_row(A)
    beta=compute_beta_row(A)
    n=A.shape[0]
    output=2*np.sqrt(n)*(1+np.log(kappa))/(1-beta)
    return output

def compute_S_B_col(B):
    kappa=compute_kappa_col(B)
    beta=compute_beta_col(B)
    n=B.shape[0]
    output=2*np.sqrt(n)*(1+np.log(kappa))/(1-beta)
    return output

def show_row(A):
    print("row stochastic matrix:")
    print("2st_eig_value:",compute_2st_eig_value(A))
    print("beta:",compute_beta_row(A))
    print("spectral gap:",1-compute_beta_row(A))
    print("kappa:",compute_kappa_row(A))
    print("S_A:",compute_S_A_row(A),"\n")

def show_col(B):
    print("column stochastic matrix:")
    print("2st_eig_value:",compute_2st_eig_value(B))
    print("beta:",compute_beta_col(B))
    print("spectral gap:",1-compute_beta_col(B))
    print("kappa:",compute_kappa_col(B))
    print("S_B:",compute_S_B_col(B),"\n")
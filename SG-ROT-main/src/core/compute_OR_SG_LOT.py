"""
Compute OR_SG_LOT using the proposed method.
"""

import numpy as np
import ot


def extract_first_block_row(C, m, n, order):
    """
    Step 1: Extract first row of blocks from the full cost matrix.
    """
    H, W = C.shape
    assert H == m * order and W == n * order
    blocks = np.zeros((order, m, n))
    for k in range(order):
        c0 = k * n
        blocks[k] = C[0:m, c0:c0 + n]
    return blocks


def build_K_from_blocks(blocks):
    """
    Step 2: Compute elementwise min (K) and argmin index (K_star) across blocks.
    """
    K = np.min(blocks, axis=0)
    K_star = np.argmin(blocks, axis=0)
    return K, K_star


def solve_small_ot(G, a, b, m, n):
    """
    Step 3: Solve small OT problem on K with reduced marginals.
    """
    a_f = a[:m]
    b_f = b[:n]
    S = ot.emd(a_f, b_f, G, numItermax=10000000)
    small_cost = np.sum(S * G)
    return S, small_cost


def build_P_list_from_S_K_star(S, K_star, order):
    """
    Step 4: Obtain solutions of OR_SG_LOT from S and K_star.
    """
    return [np.where(K_star == k, S, 0.0) for k in range(order)]


def compute_OR_SG_LOT(C, a, b, order):
    """
    Main entry: solve OR_SG_LOT transport and return small cost + P_list.
    """
    H, W = C.shape
    m = int(H/order)
    n = int(W/order)
    blocks = extract_first_block_row(C, m, n, order)
    K, K_star = build_K_from_blocks(blocks)
    S, small_cost = solve_small_ot(K, a, b, m, n)
    P_list = build_P_list_from_S_K_star(S, K_star, order)
    return small_cost, P_list
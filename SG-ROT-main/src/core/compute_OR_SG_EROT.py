"""
Compute OR_SG_EROT using the proposed sinkhorn method.
"""

import numpy as np
from src.core.compute_OR_SG_LOT import extract_first_block_row


def compute_OR_SG_EROT(C, a, b, order, reg, max_iter, tol):
    H, W = C.shape
    m = int(H / order)
    n = int(W / order)

    # Step 1: Extract first row of blocks
    blocks = extract_first_block_row(C, m, n, order)   # shape = (order, m, n)

    # Step 2: Compute kernel L = sum_b exp(-C^{1,b} / reg)
    exp_blocks = np.exp(-blocks / reg)        # shape (order, m, n)
    L = np.sum(exp_blocks, axis=0)                     # (m, n)

    # Step 3: Sinkhorn iterations on reduced problem (a_f, b_f, L)
    a_f = a[:m]
    b_f = b[:n]
    q = np.ones(n)

    for it in range(max_iter):
        Lq = L @ q
        p = a_f / (Lq + 1e-300)

        LTp = L.T @ p
        q = b_f / (LTp + 1e-300)

        if it % 10 == 0:
            err = np.linalg.norm(p * (L @ q) - a_f, 1)
            if err / np.linalg.norm(a_f, 1) < tol:
                break

    # Step 4: Reconstruct full transport blocks using p, q and exp_blocks
    P_list = []
    for k in range(order):
        P_k = np.outer(p, q) * exp_blocks[k]  # elementwise multiply
        P_list.append(P_k)

    return P_list
"""
Utility functions for block matrix operation, permutation and reconstruction.
"""

import numpy as np
import torch
from collections import defaultdict


def generate_block_circular_matrix(order):
    """
    Generate an order x order circulant matrix.
    """
    first_row = np.arange(1, order + 1)
    circ_mat = np.zeros((order, order), dtype=int)
    for i in range(order):
        circ_mat[i] = np.roll(first_row, i)
    return circ_mat


def reconstruct_P_from_permutation_matrix(order, m, n, perm_matrix, P_list):
    """
    reconstruct (m x order) x (n x order) matrix from permutation matrix.
    """
    P_array = np.stack(P_list, axis=0)
    idx_matrix = perm_matrix - 1
    # Create indexing array for one‑shot concatenation
    Big_T = P_array[idx_matrix].transpose(0,2,1,3).reshape(order*m, order*n)
    return Big_T


def get_permutation_matrix(C, m, n, order, decimals=6):
    """
    Given a block matrix C with blocks of size m×n (order×order blocks),
    compute a permutation matrix that maps row blocks to column blocks based on equality of block contents.
    """
    def get_block(i, j):
        return np.round(C[i*m:(i+1)*m, j*n:(j+1)*n], decimals=decimals)

    # Generate block signature (tuple of flattened values)
    def block_signature(block):
        return tuple(block.flatten())

    # Get reference blocks from the first row (may have duplicates)
    ref_blocks = [get_block(0, j) for j in range(order)]
    ref_sigs = [block_signature(b) for b in ref_blocks]

    # Build mapping from signature to list of column indices
    sig_to_indices = defaultdict(list)
    for idx, sig in enumerate(ref_sigs):
        sig_to_indices[sig].append(idx)

    res = np.zeros((order, order), dtype=int)
    res[0] = np.arange(1, order + 1)

    # Process remaining rows
    for i in range(1, order):
        used = np.zeros(order, dtype=bool)
        for j in range(order):
            block = get_block(i, j)
            sig = block_signature(block)
            if sig not in sig_to_indices:
                raise ValueError(f"Block at row {i+1}, column {j+1} does not appear in first row.")

            # Pick the first unused candidate index
            candidates = sig_to_indices[sig]
            for k in candidates:
                if not used[k]:
                    res[i, j] = k + 1  # convert to 1‑based
                    used[k] = True
                    break
            else:
                raise ValueError(
                    f"Block at row {i+1}, column {j+1} matches duplicate candidates but all already used."
                )

        # Verify that we used exactly order distinct blocks
        if not used.any():
            raise ValueError(f"Row {i+1} did not match any block.")

    return res


def permute_rows_and_cols_from_distance_pattern(A, decimals=6, device="cuda", batch_size=512):
    """
    permute rows and cols from distance pattern by gpu.
    """
    import torch

    if not torch.is_tensor(A):
        A = torch.tensor(A, dtype=torch.float32, device=device)
    else:
        A = A.to(device)

    if decimals > 0:
        A = torch.round(A * (10 ** decimals)) / (10 ** decimals)

    N_r, N_c = A.shape

    A_sorted = torch.empty_like(A)
    for i in range(0, N_r, batch_size):
        end = min(i + batch_size, N_r)
        A_sorted[i:end], _ = torch.sort(A[i:end], dim=1)

    _, row_inv = torch.unique(A_sorted, dim=0, return_inverse=True)

    row_order = torch.argsort(row_inv)
    row_counts = torch.bincount(row_inv)

    m = row_counts.numel()
    assert torch.all(row_counts == row_counts[0])
    G = row_counts[0].item()

    row_perm = row_order.reshape(m, G).T.reshape(-1)
    inv_row_perm = torch.argsort(row_perm)
    A_rows = A[row_perm]

    A_sorted_col = torch.empty_like(A_rows)
    for i in range(0, N_c, batch_size):
        end = min(i + batch_size, N_c)
        A_sorted_col[:, i:end], _ = torch.sort(A_rows[:, i:end], dim=0)

    _, col_inv = torch.unique(A_sorted_col.T, dim=0, return_inverse=True)

    col_order = torch.argsort(col_inv)
    col_counts = torch.bincount(col_inv)

    n = col_counts.numel()
    assert torch.all(col_counts == col_counts[0])
    assert col_counts[0].item() == G

    col_perm = col_order.reshape(n, G).T.reshape(-1)
    inv_col_perm = torch.argsort(col_perm)

    A_permuted = A_rows[:, col_perm]

    # Convert to CPU / NumPy once
    A_permuted_np = A_permuted.cpu().numpy()
    row_perm_np = row_perm.cpu().numpy()
    inv_row_perm_np = inv_row_perm.cpu().numpy()
    col_perm_np = col_perm.cpu().numpy()
    inv_col_perm_np = inv_col_perm.cpu().numpy()

    return (
        A_permuted_np,
        row_perm_np,
        inv_row_perm_np,
        col_perm_np,
        inv_col_perm_np,
        m, n, G
    )


def reconstruct_P_from_permuted_C(A, m, n, order, blocks, device = torch.device("cuda")):
    """
    Optimized version of matrix recovery from permuted blocks.
    """
    A_t = A.to(device) if torch.is_tensor(A) else torch.tensor(A, dtype=torch.float32, device=device)
    blocks_t = [b.to(device) if torch.is_tensor(b) else torch.tensor(b, dtype=torch.float32, device=device)
                for b in blocks]
    B = torch.cat(blocks_t, dim=1)

    # Pre-allocate result tensor
    A_expanded = torch.zeros_like(A_t)

    # Process in batches to manage GPU memory
    batch_size = min(32, m)

    for batch_start in range(0, m, batch_size):
        batch_end = min(batch_start + batch_size, m)
        batch_size_actual = batch_end - batch_start

        # Compute sorting indices for base rows in batch
        base_batch = A_t[batch_start:batch_end]  # (batch_size, n*order)
        sorted_base_idx = torch.argsort(base_batch, dim=1)
        inv_sorted_base_idx = torch.argsort(sorted_base_idx, dim=1)

        for k in range(order):
            row_indices = batch_start + torch.arange(batch_size_actual, device=device) + k * m
            cur_batch = A_t[row_indices]  # (batch_size, n*order)

            # Compute sorting indices for current rows
            sorted_cur_idx = torch.argsort(cur_batch, dim=1)

            # Compute permutation
            gathered = torch.gather(sorted_cur_idx, 1, inv_sorted_base_idx)
            perm_batch = torch.argsort(gathered, dim=1)

            # Fill the result matrix
            for local_idx, global_idx in enumerate(row_indices):
                A_expanded[global_idx] = B[batch_start + local_idx, perm_batch[local_idx]]

    if device.type == 'cuda':
        torch.cuda.synchronize()

    return A_expanded.cpu().numpy()
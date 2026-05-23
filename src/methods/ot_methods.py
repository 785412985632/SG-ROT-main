import numpy as np
import ot
from src.core.compute_OR_SG_LOT import (
    compute_OR_SG_LOT
)
from src.core.compute_OR_SG_EROT import (
    compute_OR_SG_EROT
)
from src.utils.matrix_utils import (
    reconstruct_P_from_permutation_matrix,
    reconstruct_P_from_permuted_C,
    permute_rows_and_cols_from_distance_pattern
)


class Method:
    def __init__(self,
                 group_order,
                 subgroup_order,
                 method,
                 ordering_type,
                 C,
                 a,
                 b,
                 reg,
                 perm_matrix
                 ):

        self.method = method
        self.group_order = group_order
        self.subgroup_order = subgroup_order
        self.ordering_type = ordering_type
        self.C = C
        self.a = a
        self.b = b
        self.reg = reg
        self.perm_matrix = perm_matrix

    def compute(self):
        if self.ordering_type == "block":
            m = self.C.shape[0] // self.group_order
            n = self.C.shape[1] // self.group_order
            cm = self.C.shape[0] // self.subgroup_order
            cn = self.C.shape[1] // self.subgroup_order
            if self.method == "LOT":
                P = ot.emd(a=self.a, b=self.b, M=self.C, numItermax=10000000)
                cost = np.sum(P * self.C)
            elif self.method == "EROT":
                P = ot.sinkhorn(a=self.a, b=self.b, M=self.C, reg=self.reg, numItermax=10000, stopThr=1e-6)
                idx = P > 0
                cost = np.sum(P * self.C) + self.reg * np.sum(P[idx] * (np.log(P[idx]) - 1))
            elif self.method == "C_LOT":
                small_cost, P_list = compute_OR_SG_LOT(C=self.C, a=self.a, b=self.b, order=self.subgroup_order)
                P = reconstruct_P_from_permutation_matrix(self.subgroup_order, cm, cn, self.perm_matrix, P_list)
                cost = small_cost * self.subgroup_order
            elif self.method == "C_EROT":
                P_list = compute_OR_SG_EROT(C=self.C, a=self.a, b=self.b, order=self.subgroup_order,
                                            reg=self.reg, max_iter=1000, tol=1e-6)
                P = reconstruct_P_from_permutation_matrix(order=self.subgroup_order, m=cm, n=cn,
                                                          perm_matrix=self.perm_matrix, P_list=P_list)
                P_row = np.hstack(P_list)
                C_row = self.C[:cm]
                idx = P_row > 0
                cost = (np.sum(P_row * C_row) * self.subgroup_order +
                        self.subgroup_order * self.reg * np.sum(P_row[idx] * (np.log(P_row[idx]) - 1)))
            elif self.method == "SG_LOT":
                small_cost, P_list = compute_OR_SG_LOT(C=self.C, a=self.a, b=self.b, order=self.group_order)
                P = reconstruct_P_from_permutation_matrix(self.group_order, m, n, self.perm_matrix, P_list)
                cost = small_cost * self.group_order
            elif self.method == "SG_EROT":
                P_list = compute_OR_SG_EROT(C=self.C, a=self.a, b=self.b, order=self.group_order,
                                            reg=self.reg, max_iter=1000, tol=1e-6)
                P = reconstruct_P_from_permutation_matrix(order=self.group_order, m=m, n=n,
                                                          perm_matrix=self.perm_matrix, P_list=P_list)
                P_row = np.hstack(P_list)
                C_row = self.C[:m]
                idx = P_row > 0
                cost = (np.sum(P_row * C_row) * self.group_order +
                        self.group_order * self.reg * np.sum(P_row[idx] * (np.log(P_row[idx]) - 1)))
            else:
                raise ValueError(f"Unknown method {self.method} for block ordering")
        elif self.ordering_type == "random":
            P_list = None
            if self.method == "LOT":
                P = ot.emd(a=self.a, b=self.b, M=self.C, numItermax=10000000)
                cost = np.sum(P * self.C)
            elif self.method == "EROT":
                P = ot.sinkhorn(a=self.a, b=self.b, M=self.C, reg=self.reg, numItermax=10000, stopThr=1e-6)
                idx = P > 0
                cost = np.sum(P * self.C) + self.reg * np.sum(P[idx] * (np.log(P[idx]) - 1))
            elif self.method in ["SG_LOT", "SG_EROT"]:
                (C_permuted,
                 row_perm,
                 inv_row_perm,
                 col_perm,
                 inv_col_perm,
                 m,
                 n,
                 group_order) = permute_rows_and_cols_from_distance_pattern(self.C)
                self.a = self.a[row_perm]
                self.b = self.b[col_perm]
                if self.method == "SG_LOT":
                    _, P_list = compute_OR_SG_LOT(C=C_permuted, a=self.a, b=self.b, order=group_order)
                elif self.method == "SG_EROT":
                    P_list = compute_OR_SG_EROT(C=C_permuted, a=self.a, b=self.b, order=group_order, reg=self.reg,
                                                max_iter=1000, tol=1e-6)
                P_after_reconstructed = reconstruct_P_from_permuted_C(C_permuted, m, n, group_order, P_list)
                P = P_after_reconstructed[inv_row_perm, :][:, inv_col_perm]
                row_P = np.hstack(P_list)
                row_C = C_permuted[:m]
                if self.method == "SG_LOT":
                    cost = np.sum(row_P * row_C) * group_order
                elif self.method == "SG_EROT":
                    idx = row_P > 0
                    cost = (np.sum(row_P * row_C) * group_order +
                        group_order * self.reg * np.sum(row_P[idx] * (np.log(row_P[idx]) - 1)))
            else:
                raise ValueError(f"Unknown method {self.method} for random ordering")
        else:
            raise ValueError(f"Unknown ordering {self.ordering_type}")
        return P, cost

"""
Main processing module for experiments.
"""

import os
import time
import numpy as np
import pandas as pd
from src.dataset.symmetry_data_generator import SymmetryDataGenerator
from src.methods.ot_methods import Method
from src.utils.utils import (
    create_distance_matrix,
    save_matrix_to_png,
    save_matrix_to_csv
)
from src.utils.matrix_utils import (
    get_permutation_matrix,
    generate_block_circular_matrix
)


def process(config):
    group_type = config["group_type"]
    group_order = config["group_order"]
    orbit_num = config["orbit_num"]
    dim = config["dim"]
    ordering_type = config["ordering_type"]
    method = config["method"]
    subgroup_order = config["subgroup_order"]
    reg = config["reg"]
    show_ordering = config["show_ordering"]

    data_folder = f"./data/{group_type}/{ordering_type}/orbit_num{orbit_num}_group_order{group_order}"
    result_folder = f"./result/{group_type}/{ordering_type}/orbit_num{orbit_num}_group_order{group_order}"
    os.makedirs(result_folder, exist_ok=True)

    source_generator = SymmetryDataGenerator(
        domain="source",
        group_type=group_type,
        group_order=group_order,
        orbit_num=orbit_num,
        dim=dim,
        ordering_type=ordering_type,
        output_folder=data_folder,
        show_ordering=show_ordering,
    )
    df_source, a = source_generator.generate()

    target_generator = SymmetryDataGenerator(
        domain="target",
        group_type=group_type,
        group_order=group_order,
        orbit_num=orbit_num,
        dim=dim,
        ordering_type=ordering_type,
        output_folder=data_folder,
        show_ordering=show_ordering,
    )
    df_target, b = target_generator.generate()

    C = create_distance_matrix(df_source, df_target, data_folder, ordering_type)

    if ordering_type == "block":
        if method in ["C_EROT", "C_LOT"]:
            perm_matrix = generate_block_circular_matrix(subgroup_order)
        elif method in ["SG_EROT", "SG_LOT"]:
            m = C.shape[0] // group_order
            n = C.shape[1] // group_order
            perm_matrix = get_permutation_matrix(C=C, m=m, n=n, order=group_order)
        else:
            perm_matrix = None
    else:
        perm_matrix = None

    ot_method = Method(
        ordering_type=ordering_type,
        method=method,
        C=C,
        a=a,
        b=b,
        reg=reg,
        group_order=group_order,
        subgroup_order=subgroup_order,
        perm_matrix=perm_matrix
    )
    start = time.perf_counter()
    P, cost = ot_method.compute()
    time_curr = time.perf_counter() - start

    img_path = os.path.join(result_folder, f"{method}_P.png")
    save_matrix_to_png(P, save_path=img_path)

    if method in ["LOT", "EROT"]:
        csv_path = os.path.join(result_folder, f"{method}_P.csv")
        save_matrix_to_csv(P, save_path=csv_path)
        matrix_distance = 0.0
    else:
        if method.startswith("C_"):
            original_method = method[2:]
        elif method.startswith("SG_"):
            original_method = method[3:]
        else:
            raise ValueError(f"Unknown method: {method}")

        orig_csv_path = os.path.join(result_folder, f"{original_method}_P.csv")
        if not os.path.exists(orig_csv_path):
            raise FileNotFoundError(f"Run {original_method} first to generate {orig_csv_path}")
        P_orig = pd.read_csv(orig_csv_path, header=None).values
        matrix_distance = np.linalg.norm(P - P_orig, 'fro')

    info_path = os.path.join(result_folder, f"{method}_info.csv")
    record = pd.DataFrame([{
        "method": method,
        "cost": cost,
        "time": time_curr,
        "matrix_distance": matrix_distance
    }])
    record.to_csv(info_path, index=False)

    print(f"✓ {method:10s} | cost={cost:.6f} | time={time_curr:.4f}s | distance={matrix_distance:.2e}")
"""
Basic utility functions for experiments.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def create_distance_matrix(df_source, df_target, save_folder, ordering_type):
    os.makedirs(save_folder, exist_ok=True)
    csv_path = os.path.join(save_folder, f"{ordering_type}_C.csv")
    png_path = os.path.join(save_folder, f"{ordering_type}_C.png")
    if os.path.exists(csv_path) and os.path.exists(png_path):
        print(f"⚡ Distance matrix already exists, loading: \n  {csv_path}")
        C = pd.read_csv(csv_path, header=None).values
        return C

    source_coords = df_source.iloc[:, 1:].values  # exclude id column
    target_coords = df_target.iloc[:, 1:].values
    C = cdist(source_coords, target_coords, metric='euclidean')
    save_matrix_to_png(M=C, cmap='viridis', save_path=png_path)
    save_matrix_to_csv(M=C, save_path=csv_path)
    return C


def save_matrix_to_csv(M, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savetxt(save_path, M, delimiter=',')
    print(f"CSV saved: {save_path}")


def save_matrix_to_png(M, cmap="viridis", save_path=None, vmax=None, vmin=0):
    plt.figure(figsize=(4, 4))

    if vmax is None:
        vmax = M.max()

    plt.imshow(M, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.tight_layout(pad=0.0)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.0
        )
        print(f"PNG saved: {save_path}")

    plt.close()
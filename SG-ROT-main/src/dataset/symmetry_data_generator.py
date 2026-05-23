"""
Data generator for symmetric point clouds under finite group actions.
Supports groups: Z_n (cyclic), D_l (dihedral), T (tetrahedral), O (octahedral), I (icosahedral).
Generates Gaussian blobs and applies group transformations to create structured point sets.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.cm as cm


class SymmetryDataGenerator:
    """
    Generate point clouds with symmetry under a given finite group.
    The points are created by applying group transformations to a Gaussian seed cloud.
    Optionally shuffles the ordering of points and their associated weights.
    """

    def __init__(self,
                 domain,
                 group_type,
                 group_order,
                 orbit_num,
                 dim,
                 ordering_type,
                 show_ordering,
                 output_folder):
        """
        Args:
            domain (str): 'source' or 'target' – affects color map.
            group_type (str): 'Z', 'D', 'T', 'O', 'I'.
            group_order (int): Order of the group (e.g., 6 for D_3).
            orbit_num (int): Number of Gaussian blobs (orbits).
            dim (int): 2 or 3.
            ordering_type (str): 'Block' or 'Random' – controls shuffling.
            show_ordering (bool): Whether to annotate point IDs in the plot.
            output_folder (str): Directory to save CSV files and plots.
        """
        self.domain = domain
        self.group_type = group_type
        self.group_order = group_order
        self.l = group_order // 2          # Used for dihedral groups
        self.orbit_num = orbit_num
        self.dim = dim
        self.center = np.ones(dim)
        self.ordering_type = ordering_type
        self.show_ordering = show_ordering
        self.output_folder = output_folder

        # Dispatch table for transformation generators
        self.group_transforms = {
            "Z": self._transforms_Z,
            "D": self._transforms_D,
            "T": self._transforms_T,
            "O": self._transforms_O,
            "I": self._transforms_I
        }
        if self.group_type not in self.group_transforms:
            raise ValueError(f"Unknown group_type: {self.group_type}")

        os.makedirs(self.output_folder, exist_ok=True)

    def generate(self):
        """
        Generate point cloud and weight vector w.
        If CSV files already exist, they are loaded directly.
        Returns:
            df (pd.DataFrame): Points with columns [id, x1, ..., xd].
            w (np.ndarray): Normalized weight vector.
        """
        points_csv = os.path.join(
            self.output_folder, f"{self.ordering_type}_{self.domain}_points.csv"
        )
        w_csv = os.path.join(
            self.output_folder, f"{self.ordering_type}_{self.domain}_w.csv"
        )

        # ----- Load existing data if available -----
        if os.path.exists(points_csv) and os.path.exists(w_csv):
            print(f"⚡ {self.domain} data already exists, loading:\n  {points_csv}\n  {w_csv}")
            df = pd.read_csv(points_csv)
            w = pd.read_csv(w_csv)["w"].to_numpy()
            return df, w

        # ----- Step 1: Generate seed Gaussian cloud -----
        X0 = self._generate_gaussian_cloud()

        # ----- Choose colormap based on domain -----
        if self.domain.lower() == "source":
            cmap = cm.get_cmap("autumn")   # warm colors
        elif self.domain.lower() == "target":
            cmap = cm.get_cmap("winter")   # cool colors
        else:
            raise ValueError("domain must be 'source' or 'target'")

        # Each orbit gets a distinct color; repeat for each group element
        base_colors = cmap(np.linspace(0.15, 0.85, self.orbit_num))
        colors = np.tile(base_colors, (self.group_order, 1))

        # ----- Step 2: Obtain all group transformations -----
        transforms = self.group_transforms[self.group_type]()

        # ----- Step 3: Apply transformations to generate all points -----
        all_points = []
        id_counter = 1
        for H in transforms:
            X_trans = self._apply_homogeneous_transform(X0, H)
            for pt in X_trans:
                all_points.append([id_counter, *pt])
                id_counter += 1

        cols = ["id"] + [f"x{i + 1}" for i in range(self.dim)]
        df = pd.DataFrame(all_points, columns=cols)

        # ----- Step 4: Generate weight vector w (small random perturbation) -----
        eps = 0.05
        v_base = 1.0 + eps * np.random.randn(self.orbit_num)
        v_base = np.clip(v_base, 1e-6, None)
        w = np.tile(v_base, self.group_order)
        w = w / np.sum(w)

        # ----- Step 5: Optionally shuffle ordering (Random ordering) -----
        if self.ordering_type == "random":
            rng = np.random.RandomState(42)
            perm = rng.permutation(len(df))

            coord_cols = df.columns[1:]
            df[coord_cols] = df.loc[perm, coord_cols].reset_index(drop=True)
            w = w[perm]
            colors = colors[perm]

        # ----- Save to CSV -----
        df.to_csv(points_csv, index=False, float_format="%.8f")
        pd.DataFrame({"w": w}).to_csv(w_csv, index=False, float_format="%.12f")

        # ----- Create and save a plot -----
        self._save_plot(df, colors, f"{self.ordering_type}_{self.domain}_points")

        return df, w

    # -------------------- Gaussian cloud generation --------------------
    def _generate_gaussian_cloud(self):
        """
        Generate a set of Gaussian blobs around the center.
        """
        cov = np.eye(self.dim) * 0.5
        return np.random.multivariate_normal(mean=self.center, cov=cov, size=self.orbit_num)

    # -------------------- Homogeneous transformation utilities --------------------
    @staticmethod
    def _apply_homogeneous_transform(X, H):
        """
        Apply a 4x4 homogeneous transformation matrix H (or 3x3 for 2D) to points X.
        X: (n, dim) array.
        Returns transformed points (n, dim).
        """
        n, dim = X.shape
        X_h = np.hstack([X, np.ones((n, 1))])   # to homogeneous coordinates
        X_trans = (H @ X_h.T).T[:, :dim]        # back to Cartesian
        return X_trans

    # -------------------- Cyclic group Z_n (2D) --------------------
    def _transforms_Z(self):
        """
        Return list of rotation matrices for cyclic group Z_n.
        """
        H_list = []
        for k in range(self.group_order):
            theta = 2 * np.pi * k / self.group_order
            R = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta),  np.cos(theta)]])
            H = np.eye(3)
            H[:2, :2] = R
            H_list.append(H)
        return H_list

    # -------------------- Dihedral group D_l (2D) --------------------
    def _transforms_D(self):
        """
        Return list of rotation and reflection matrices for dihedral group D_l.
        """
        if self.l is None:
            raise ValueError("Please set l (group_order//2) for dihedral group.")
        H_list = []
        for k in range(self.l):
            theta = 2 * np.pi * k / self.l
            R = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta),  np.cos(theta)]])
            H = np.eye(3)
            H[:2, :2] = R
            H_list.append(H)

            # Mirror reflection along the x-axis (after rotation)
            M = np.array([[-1, 0], [0, 1]])
            H_mirror = np.eye(3)
            H_mirror[:2, :2] = R @ M
            H_list.append(H_mirror)
        return H_list

    # -------------------- Tetrahedral group T (3D) --------------------
    def _transforms_T(self):
        """
        Return 12 rotation matrices for the tetrahedral group T (order 12).
        """
        H_list = []

        # Identity
        H_list.append(np.eye(4))

        # 120° and 240° rotations about axes through vertices and opposite face centers
        axes = [
            [1, 1, 1],
            [-1, -1, 1],
            [-1, 1, -1],
            [1, -1, -1]
        ]
        for axis in axes:
            for theta in [2 * np.pi / 3, 4 * np.pi / 3]:
                R = self._rotation_matrix_3d(axis, theta)
                H = np.eye(4)
                H[:3, :3] = R
                H_list.append(H)

        # 180° rotations about axes through midpoints of opposite edges
        axes_180 = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
        for axis in axes_180:
            R = self._rotation_matrix_3d(axis, np.pi)
            H = np.eye(4)
            H[:3, :3] = R
            H_list.append(H)

        # Reorder to the max cyclic subgroup structure
        new_order = [0, 3, 10, 8, 1, 6, 7, 11, 2, 9, 4, 5]
        H_list = [H_list[i] for i in new_order]
        return H_list

    # -------------------- Octahedral group O (3D) --------------------
    def _transforms_O(self):
        """
        Return 24 rotation matrices for the octahedral group O (order 24).
        """
        H_list = []

        # Identity
        H_list.append(np.eye(4))

        # Rotations about coordinate axes: 90°, 180°, 270°
        axes = np.eye(3)   # x, y, z
        angles = [np.pi / 2, np.pi, 3 * np.pi / 2]
        for axis in axes:
            for theta in angles:
                R = self._rotation_matrix_3d(axis, theta)
                H = np.eye(4)
                H[:3, :3] = R
                H_list.append(H)

        # Rotations about body diagonals (±120°, ±240°)
        diag_axes = np.array([
            [1, 1, 1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, -1]
        ], dtype=float)
        for axis in diag_axes:
            axis = axis / np.linalg.norm(axis)
            for theta in [2 * np.pi / 3, 4 * np.pi / 3]:
                R = self._rotation_matrix_3d(axis, theta)
                H = np.eye(4)
                H[:3, :3] = R
                H_list.append(H)

        # 180° rotations about axes through centers of opposite faces (swap two axes)
        face_axes = [
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [-1, 1, 0],
            [-1, 0, 1],
            [0, -1, 1]
        ]
        for axis in face_axes:
            axis = axis / np.linalg.norm(axis)
            R = self._rotation_matrix_3d(axis, np.pi)
            H = np.eye(4)
            H[:3, :3] = R
            H_list.append(H)

        # Reorder to the max cyclic subgroup structure
        new_order = [10, 0, 11, 4, 18, 3,
                     20, 7, 6, 12, 5, 17,
                     13, 8, 14, 22, 21, 23,
                     1, 9, 19, 16, 2, 15]
        H_list = [H_list[i] for i in new_order]
        return H_list

    # -------------------- Icosahedral group I (3D) --------------------
    def _transforms_I(self):
        """
        Return 60 rotation matrices for the icosahedral group I (order 60).
        """
        phi = (1 + np.sqrt(5)) / 2   # golden ratio

        def R(axis, theta):
            """Rodrigues rotation formula."""
            axis = np.array(axis, float)
            axis /= np.linalg.norm(axis)
            x, y, z = axis
            c = np.cos(theta)
            s = np.sin(theta)
            C = 1 - c
            return np.array([
                [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
                [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
                [z*x*C - y*s,   z*y*C + x*s, c + z*z*C]
            ])

        def to_H(R3):
            """Convert 3x3 rotation to 4x4 homogeneous matrix."""
            H = np.eye(4)
            H[:3, :3] = R3
            return H

        def unique_axes(axes):
            """Return normalized axes, removing duplicates and opposites."""
            out = []
            for a in axes:
                a = np.array(a, float)
                a /= np.linalg.norm(a)
                if not any(np.allclose(a, b) or np.allclose(a, -b) for b in out):
                    out.append(a)
            return out

        # Fivefold axes (12 axes)
        five_axes_raw = [
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ]
        five_axes = unique_axes(five_axes_raw)

        # Build generators: 72°, 144°, 216°, 288° rotations around fivefold axes
        M = []
        for a in five_axes:
            for k in [1, 2, 3, 4]:
                M.append(to_H(R(a, 2 * np.pi * k / 5)))

        def compose(indices):
            """Compose several generator matrices (order matters: H = M[i] @ ...)."""
            H = np.eye(4)
            for i in indices:
                H = M[i] @ H
            return H

        # Pre‑selected compositions to obtain exactly 60 distinct rotations
        selected_indices_compositions = [
            [12], [13], [14], [15],
            [0], [0, 12], [0, 13], [0, 14], [0, 15],
            [1], [1, 12], [1, 13], [1, 14], [1, 15],
            [2], [2, 12], [2, 13], [2, 14], [2, 15],
            [3], [3, 12], [3, 13], [3, 14], [3, 15],
            [8], [8, 12], [8, 13], [8, 14], [8, 15],
            [9], [9, 12], [9, 13], [9, 14], [9, 15],
            [11], [11, 12], [11, 13], [11, 14], [11, 15],
            [17], [17, 12], [17, 13], [17, 14], [17, 15],
            [18], [18, 12], [18, 13], [18, 14], [18, 15],
            [23], [23, 12], [23, 13], [23, 14], [23, 15],
            [0, 4, 16, 20], [0, 4, 16, 20, 12], [0, 4, 16, 20, 13],
            [0, 4, 16, 20, 14], [0, 4, 16, 20, 15]
        ]

        # Build final list including identity
        transforms = [np.eye(4)] + [compose(indices) for indices in selected_indices_compositions]

        # Reorder to the max cyclic subgroup structure
        new_order = [0,7,39,20,51,26,12,16,49,43,31,58,
                     1,8,35,21,52,27,13,17,45,44,32,59,
                     2,9,36,22,53,28,14,18,46,40,33,55,
                     3,5,37,23,54,29,10,19,47,41,34,56,
                     4,6,38,24,50,25,11,15,48,42,30,57]
        transforms = [transforms[i] for i in new_order]
        assert len(transforms) == 60, f"Expected 60 elements, got {len(transforms)}"
        return transforms

    @staticmethod
    def _rotation_matrix_3d(axis, theta):
        """
        Generate a 3x3 rotation matrix using Rodrigues' formula.
        axis: 3D unit vector (or any vector that will be normalized).
        theta: rotation angle in radians.
        """
        axis = np.array(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        c = np.cos(theta)
        s = np.sin(theta)
        C = 1 - c
        R = np.array([
            [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
            [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
            [z*x*C - y*s,   z*y*C + x*s, c + z*z*C]
        ])
        return R

    def _save_plot(self, df, colors, title):
        """
        Create a 2D or 3D scatter plot with a bounding box and optional point IDs.
        Saves the plot as a PNG file.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
        import os

        plt.rcParams.update({
            "figure.dpi": 300,
            "font.size": 10,
            "axes.linewidth": 0.8,
        })

        fig = plt.figure(figsize=(4, 4))

        if self.dim == 2:
            ax = fig.add_subplot(111)
            ax.scatter(
                df["x1"], df["x2"],
                c=colors,
                s=30,
                alpha=0.9,
                edgecolors="none"
            )

            if self.show_ordering:
                for _, row in df.iterrows():
                    ax.text(row["x1"], row["x2"], str(int(row["id"])),
                            fontsize=7, ha='right', va='bottom')

            # Center at origin with padding
            all_x = df["x1"].to_numpy()
            all_y = df["x2"].to_numpy()
            max_abs = max(np.abs(all_x).max(), np.abs(all_y).max())
            padding = 0.15 * max_abs
            lim = max_abs + padding
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal", adjustable="box")

            # Draw bounding box
            rect = patches.Rectangle(
                xy=(-lim, -lim),
                width=2 * lim,
                height=2 * lim,
                linewidth=0.8,
                edgecolor='black',
                facecolor='none'
            )
            ax.add_patch(rect)

            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])

        elif self.dim == 3:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(
                df["x1"], df["x2"], df["x3"],
                c=colors,
                s=30,
                alpha=0.9,
                depthshade=False
            )

            if self.show_ordering:
                for _, row in df.iterrows():
                    ax.text(row["x1"], row["x2"], row["x3"], str(int(row["id"])),
                            fontsize=7)

            # Center at origin with padding
            all_coords = df.iloc[:, 1:].to_numpy()
            mins = all_coords.min(axis=0)
            maxs = all_coords.max(axis=0)
            centers = (maxs + mins) / 2
            max_range = np.max(maxs - mins) / 2 * 1.15
            ax.set_xlim(centers[0] - max_range, centers[0] + max_range)
            ax.set_ylim(centers[1] - max_range, centers[1] + max_range)
            ax.set_zlim(centers[2] - max_range, centers[2] + max_range)

            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            # Draw 3D bounding box (12 edges)
            X = [centers[0] - max_range, centers[0] + max_range]
            Y = [centers[1] - max_range, centers[1] + max_range]
            Z = [centers[2] - max_range, centers[2] + max_range]
            for x in X:
                for y in Y:
                    ax.plot([x, x], [y, y], Z, color='black', linewidth=0.8)
            for x in X:
                for z in Z:
                    ax.plot([x, x], Y, [z, z], color='black', linewidth=0.8)
            for y in Y:
                for z in Z:
                    ax.plot(X, [y, y], [z, z], color='black', linewidth=0.8)

        # Save plot
        os.makedirs(self.output_folder, exist_ok=True)
        save_path = os.path.join(self.output_folder, f"{title}.png")
        plt.tight_layout(pad=0.2)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"PNG saved: {save_path}")
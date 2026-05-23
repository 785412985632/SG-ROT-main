"""
Configuration file generator for experiments.

This script creates YAML configuration files for various experimental setups.
It supports:
    - Group types: D (active), Z, T, O, I (others commented out)
    - Ordering types: Block, Random
    - Methods: depend on ordering type.
        * Block ordering: C_LOT, SG_LOT, LOT, C_EROT, SG_EROT, EROT
        * Random ordering: SG_LOT, LOT, SG_EROT, EROT

Directory structure generated:
    ./{group_type}/{ordering_type}/orbit_num{orbit_num}_group_order{group_order}/{method}_config.yaml
"""

import os
import yaml


def create_config_files():
    """
    Main function that iterates over all parameter combinations and writes config.yaml files.
    """

    # List of group types
    group_types = [
        # 'Z',
        'D',
        # 'T',
        # 'O',
        # 'I'
    ]

    # Order of groups
    group_order_map = {
        'Z': [3],      # any number
        'D': [6],      # any even number
        'T': [12],     # fixed 12
        'O': [24],     # fixed 24
        'I': [60]      # fixed 60
    }

    # Dimensionality of the representation
    dim_map = {
        'Z': 2,
        'D': 2,
        'T': 3,
        'O': 3,
        'I': 3,
    }

    # Number of orbits
    orbit_num_map = {
        'Z': [1600],
        'D': [
            800,
            5
        ],
        'T': [400],
        'O': [200],
        'I': [80],
    }

    ordering_types = [
        'block',
        'random'
    ]
    # Two kinds of ordering

    def get_methods(ordering_type):
        """
        Return a list of method names for the given ordering_type.
        """
        if ordering_type == 'block':
            return ['C_LOT', 'SG_LOT', 'LOT', 'C_EROT', 'SG_EROT', 'EROT']
        else:  # Random
            return ['SG_LOT', 'LOT', 'SG_EROT', 'EROT']

    def get_subgroup_order(group_type, group_order):
        """
        Return the order of max cyclic subgroups of the given group.
        """
        if group_type == 'Z':
            return group_order
        elif group_type == 'D':
            return group_order // 2
        elif group_type == 'T':
            return 3
        elif group_type == 'O':
            return 4
        elif group_type == 'I':
            return 5
        return None

    def get_reg(method):
        """
        Return regularisation based on method.
        """
        return 0.1 if method.endswith('EROT') else 0

    # ----- Main nested loops over all parameter combinations -----
    for group_type in group_types:
        for orbit_num in orbit_num_map[group_type]:
            for group_order in group_order_map[group_type]:
                for ordering_type in ordering_types:
                    methods = get_methods(ordering_type)
                    for method in methods:

                        file_path = (
                            f"./config/{group_type}/{ordering_type}/"
                            f"orbit_num{orbit_num}_group_order{group_order}/{method}_config.yaml"
                        )
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)

                        dim = dim_map[group_type]
                        subgroup_order = get_subgroup_order(group_type, group_order)
                        reg = get_reg(method)

                        config = {
                            'paths': {
                                'data_root': '../data',
                                'result_root': '../result'
                            },
                            'group_type': group_type,
                            'ordering_type': ordering_type,
                            'method': method,
                            'orbit_num': orbit_num,
                            'group_order': group_order,
                            'dim': dim,
                            'subgroup_order': subgroup_order,
                            'reg': reg,
                            'show_ordering': False,  # show id of points
                        }

                        with open(file_path, 'w') as f:
                            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                        print(f"Created: {file_path}")

if __name__ == "__main__":
    create_config_files()
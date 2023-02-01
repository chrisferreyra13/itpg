"""Useful functions to manage input data for analysis."""
# Authors: Christian Ferreyra, chrisferreyra13@gmail.com
# Date: 2022

import os


def get_input_filepath(dataset_name, filename, return_dataset=False):
    """Return input filepath."""

    dataset_path = os.path.join(os.getcwd(), 'data', dataset_name)

    if not os.path.exists(dataset_path):
        raise ValueError('Dataset does not exist')

    filepath = os.path.join(dataset_path, filename)

    if not os.path.exists(filepath):
        raise ValueError('Input file does not exist')

    if return_dataset:
        return filepath, dataset_path

    return filepath

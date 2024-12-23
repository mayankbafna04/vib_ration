# utils.py

import os

def check_file_exists(file_path):
    """
    Check if a file exists at the given path.
    Args:
        file_path (str): The path to check.
    Returns:
        bool: True if file exists, False otherwise.
    """
    return os.path.isfile(file_path)

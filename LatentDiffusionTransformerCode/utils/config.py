"""
YAML/JSON configuration loading utility.
"""
import yaml
from typing import Dict

def load_config(config_path: str) -> Dict:
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): The path to the YAML config file.

    Returns:
        Dict: A dictionary containing the configuration.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

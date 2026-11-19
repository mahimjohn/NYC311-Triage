
from pathlib import Path
import os

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """
    Load the YAML configuration file and expand environment variables.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")

    return _expand_environment_variables(config)


def _expand_environment_variables(value):
    """Recursively expand environment variables in configuration values."""
    if isinstance(value, dict):
        return {
            key: _expand_environment_variables(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_expand_environment_variables(item) for item in value]

    if isinstance(value, str):
        return os.path.expandvars(value)

    return value


def get_config_value(config: dict, key_path: str, default=None):
    """
    Retrieve a nested configuration value using dot notation.

    Example:
        get_config_value(config, "ingestion.page_size")
    """
    value = config

    for key in key_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default

        value = value[key]

    return value

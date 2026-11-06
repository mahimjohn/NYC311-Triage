
import yaml
from pathlib import Path


def get_project_root():
    return Path(__file__).resolve().parents[2]


def load_config(config_path=None):
    """
    Load project configuration from a YAML file.

    Args:
        config_path: Optional custom path to a YAML configuration file.

    Returns:
        dict: Parsed configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the YAML file contains invalid syntax.
    """

    if config_path is None:
        config_path = get_project_root() / "config" / "config.yaml"
    else:
        config_path = Path(config_path).expanduser()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    try:
        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

    except yaml.YAMLError as exc:
        raise ValueError(
            f"Invalid YAML configuration file: {config_path}"
        ) from exc

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration must contain a YAML dictionary: {config_path}"
        )

    return config

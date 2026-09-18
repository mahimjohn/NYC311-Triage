import yaml
from pathlib import Path


def get_project_root():
    return Path(__file__).resolve().parents[2]

def load_config():
    with open(get_project_root() / "config" / "config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config
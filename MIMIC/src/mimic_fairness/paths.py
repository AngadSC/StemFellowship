from pathlib import Path
import yaml


def load_config(config_path: str | None = None) -> dict:
    if config_path is None:
        config_path = project_root() / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_path(*parts: str) -> Path:
    return project_root().joinpath(*parts)


``python
import yaml
from pathlib import Path
import pytest

CONFIG_PATH = Path("configs/config.yaml")

def test_config_exists():
    assert CONFIG_PATH.exists(), "config.yaml not found!"

def test_required_keys():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    required_keys = ["directories", "flags", "video_processing", "pose_extraction"]
    for key in required_keys:
        assert key in config, f"Missing '{key}' in config.yaml"

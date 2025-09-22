from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]  # project root (adjust if needed)
CONFIG_DIR = ROOT / "configs"

def load_yaml(name: str):
    p = CONFIG_DIR / name
    with open(p, "r") as f:
        return yaml.safe_load(f)

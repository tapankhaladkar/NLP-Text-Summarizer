from pathlib import Path

# Get the project root (going up from constants/__init__.py to project root)
# constants/__init__.py -> textsummarizer -> src -> project_root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

CONFIG_FILE_PATH = PROJECT_ROOT / "config" / "config.yaml"
PARAMS_FILE_PATH = PROJECT_ROOT / "params.yaml"
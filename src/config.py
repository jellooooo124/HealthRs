import os
import yaml

"""
Config loader: reads YAML config files under configs/ and makes them available to other modules.

Two environment variables switch where data and results live (used by scripts/reproduce.sh):
    HEALTHREC_DATA_CONFIG  name of the data config to use instead of configs/data.yaml (e.g. data_sample)
    HEALTHREC_OUTPUTS      results directory instead of outputs/ (relative to the project root or absolute)
"""

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONF_DIR = os.path.join(_PROJECT_ROOT, "configs")

# Autoload .env file from project root if it exists
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
except ImportError:
    pass  # python-dotenv not installed, just use system env vars

# Cache to avoid re-reading files every time
_configs: dict = {}


def get_project_root() -> str:
    return _PROJECT_ROOT


def get_outputs_dir(*parts: str) -> str:
    """Results directory (outputs/ by default, or $HEALTHREC_OUTPUTS), joined with `parts`."""
    base = os.environ.get("HEALTHREC_OUTPUTS", "outputs")
    return os.path.join(base if os.path.isabs(base) else os.path.join(_PROJECT_ROOT, base), *parts)


def load_config(name: str) -> dict:
    """Load configs/<name>.yaml (cached). `data` can be redirected with $HEALTHREC_DATA_CONFIG."""
    if name == "data":
        name = os.environ.get("HEALTHREC_DATA_CONFIG", "data")
    if name not in _configs:
        path = os.path.join(_CONF_DIR, f"{name}.yaml")
        with open(path, "r", encoding="utf-8") as f:
            _configs[name] = yaml.safe_load(f)
    return _configs[name]


def get_agent_config() -> dict:
    return load_config("agent_config")


def get_prompt_config() -> dict:
    return load_config("prompts")

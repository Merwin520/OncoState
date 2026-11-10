"""Package-relative paths and configuration. No absolute paths are stored in the package."""
import os
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parent
CONFIGS = PKG / 'configs'
EXPORTS = PKG / 'exports'
STATES = PKG / 'patient_states'
LOGS = PKG / 'logs'


def config(name='experiment'):
    return yaml.safe_load((CONFIGS / f'{name}.yaml').read_text())


def work_dir():
    """Large or temporary files (IDC index slices, source spreadsheets, imaging cache) live outside the package."""
    c = config()
    p = Path(os.environ.get(c['work_dir_env']) or (PKG / c['work_dir_default'])).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def metadata_dir():
    p = work_dir() / 'metadata'
    p.mkdir(exist_ok=True)
    return p


def source_dir():
    p = metadata_dir() / 'source'
    p.mkdir(exist_ok=True)
    return p


def cache_dir():
    p = work_dir() / 'cache'
    p.mkdir(exist_ok=True)
    return p

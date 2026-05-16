import os
import yaml


CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'signal_config.yaml')

_config_cache = None
_config_mtime = None


def load_signal_config():
    global _config_cache, _config_mtime
    try:
        mtime = os.path.getmtime(CONFIG_PATH)
    except OSError:
        mtime = None

    if _config_cache is not None and _config_mtime == mtime:
        return _config_cache

    with open(CONFIG_PATH, 'r') as f:
        _config_cache = yaml.safe_load(f)
    _config_mtime = mtime
    return _config_cache

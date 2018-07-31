"""Load, save and update json files."""
import os
import io
import json
from datetime import datetime


def save_json(filename, config):
    """Save configuration file as JSON.

    Parameters
    ----------
    filename : string
        Name of the configuration file to save.
    config : dict
        Dictionary of arguments to save.
    """
    # Ensure Python version compatibility
    try:
        to_unicode = unicode
    except NameError:
        to_unicode = str

    if filename:
        with io.open(filename, 'w', encoding='utf8') as f:
            str_ = json.dumps(config, indent=4, sort_keys=True,
                              separators=(',', ': '),  # Pretty printing
                              ensure_ascii=False)
            f.write(to_unicode(str_))


def load_json(filename):
    """Load configuration file as JSON.

    Parameters
    ----------
    filename : string
        Name of the configuration file to load.

    Returns
    -------
    config : dict
        Dictionary of config.
    """
    with open(filename) as f:
        # Load the configuration file :
        config = json.load(f)
    return config


def update_json(filename, update, backup=None):
    """Update a json file.

    Parameters
    ----------
    filename : str
        Full path to the json file.
    update : dict
        Dict for update.
    backup : str | None
        Backup folder if needed.
    """
    assert isinstance(update, dict)
    assert os.path.isfile(filename)
    config = load_json(filename)
    _backup_json(filename, backup)
    config.update(update)
    save_json(filename, config)


def _backup_json(filename, backup=None):
    if isinstance(backup, str):
        assert os.path.isfile(filename)
        assert os.path.exists(backup)
        # Load json file :
        config = load_json(filename)
        config_backup = config.copy()
        # Datetime :
        now = datetime.now()
        now_lst = [now.year, now.month, now.day, now.hour, now.minute,
                   now.second]
        now_lst = '_'.join([str(k) for k in now_lst])
        file, ext = os.path.splitext(os.path.split(filename)[1])
        file += now_lst + ext
        save_json(os.path.join(backup, file), config_backup)

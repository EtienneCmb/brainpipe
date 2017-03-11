import os
import brainpipe
import pickle
from shutil import rmtree
import numpy as n
from scipy.io import loadmat

__all__ = ['_bpfolders',
           '_check_bpsettings_exist',
           '_add_bpsettings_entry',
           '_path_bpsettings',
           '_load_bpsettings',
           '_save_bpsettings',
           '_check_study_exist',
           '_update_bpsettings'
           ]


def _bpfolders(directory):
    """Check if a folder exist otherwise, create it
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def _check_bpsettings_exist():
    """Check if the bpsettings file exist otherwise create it
    """
    bpPath = _path_bpsettings()
    try:
        _load_bpsettings()
    except:
        with open(bpPath, 'wb') as f:
            pickle.dump({}, f)


def _add_bpsettings_entry(name, path, creation):
    """Add a study to the bpsettings file. bpsettings is a
    dictionary organized by name of study and containing path.
    """
    bpsettings = _load_bpsettings()
    bpsettings[name] = {'path': os.path.join(path, name),
                        'created': creation
                        }
    _save_bpsettings(bpsettings)


def _path_bpsettings():
    """Get the path of bpsettings
    """
    # bpPath = os.path.dirname(brainpipe.__file__)
    return os.path.join(os.path.dirname(
        brainpipe.__file__), 'bpsettings.pickle')


def _load_bpsettings():
    """Load the bpsettings file
    """
    bpCfg = _path_bpsettings()
    with open(bpCfg, "rb") as f:
        bpsettings = pickle.load(f)
    return bpsettings


def _save_bpsettings(bpsettings):
    """Update bpsettings. The parameter should be a
    dictionary organized as bpsettings.
    """
    bpCfg = _path_bpsettings()
    with open(bpCfg, 'wb') as f:
        pickle.dump(bpsettings, f)


def _check_study_exist(self):
    """Check if a study exist
    """
    try:
        bpsettings = _load_bpsettings()
        self.path = bpsettings[self.name]['path']
        self.created = bpsettings[self.name]['created']
        print('-> '+self.name+' loaded')
    except:
        pass


def _update_bpsettings():
    """Update bpsettings list of study by checking each folder.
    """
    bpsettings = _load_bpsettings()
    bpsettingsNew = {}
    for k, i in zip(bpsettings.keys(), bpsettings.values()):
        if os.path.exists(i['path']):
            bpsettingsNew[k] = i
    _save_bpsettings(bpsettingsNew)

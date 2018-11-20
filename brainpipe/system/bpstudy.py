"""Handle brainpipe studies (files and folders)."""
import os
import re
import logging

import numpy as np
from itertools import product
from datetime import datetime

from shutil import rmtree
import matplotlib.pyplot as plt

from .logging import set_log_level
from ..io import (load_json, save_json, update_json, load_file, save_file,
                  safety_save)


BP_FILE = 'bpsettings.json'

logger = logging.getLogger('brainpipe')


class Study(object):
    """Create and manage a study with a files database.

    Parameters
    ----------
    name: string | None
        Name of the study. If this study already exists, this will load
        the path with the associated database.

    Examples
    --------
    >>> # Define variables:
    >>> path = '/home/Documents/database'
    >>> studyName = 'MyStudy'

    >>> # Create a study object :
    >>> studObj = study(name=studyName)
    >>> studObj.add(path)   # Create the study
    >>> studObj.studies()   # Print the list of studies

    >>> # Manage files in your study :
    >>> fileList = studObj.search('filter1', 'filter2', folder='features')
    >>> # Let say that fileList contain two files : ['f_1.mat', 'f_2.pickle']
    >>> # Load the second file :
    >>> data = studObj.load('features', 'file2.pickle')
    """

    def __init__(self, name, verbose=None):  # noqa
        set_log_level(verbose)
        assert isinstance(name, str)
        self.name = name
        # Get path to the bp file :
        bp_path = self._path_bpsettings()
        # If it doesn't exist, create it
        if not os.path.isfile(bp_path):
            logger.info('Brainpipe file added to the path %s' % bp_path)
            save_json(bp_path, {})
        # Check if the study exist :
        cfg = load_json(bp_path)
        if self.name not in cfg.keys():
            logger.warning("Study %s doesn't exist. Use `add` to create "
                           "it." % self.name)
            return None
        # Load the study and get path to it :
        self.config = cfg
        self.path = self['path']
        self.created = self['created']
        logger.info('Study %s loaded' % self.name)

    def __str__(self):
        """String representation."""
        st = "Study : %s\nPath : %s\nCreation : %s" % (self.name, self.path,
                                                       self.created)
        return st

    def __getitem__(self, name):
        """Get the item in the config file of the study."""
        return self.config[self.name][name]

    def __setitem__(self, name, value):
        """Set an item in the config file of the study."""
        self.config[self.name][name] = value

    # -------------------------------------------------------------
    # Manage Files:
    # -------------------------------------------------------------
    def add(self, path):
        """Add a new study.

        Once the study is added, the folowing structure is created :

            * name: the root folder of the study. Same name as the study
            * /anatomy: physiological informations
            * /backup: backup files
            * /classified: classified features
            * /database: datasets of the study
            * /feature: features extracted from the diffrents datasets
            * /figure: figures of the study
            * /multifeature: multifeatures files
            * /other: any other kind of files
            * /setting: study settings

        Parameters
        ----------
        path: string
            Path to the study.
        """
        assert os.path.isdir(path)
        # Load the config :
        bp_path = self._path_bpsettings()
        self.config = load_json(bp_path)
        if self.name in self.config.keys():
            logger.warning("%s already exist. Use a different name or "
                           "delete it before" % self.name)
            return None
        # Main studyName directory :
        self._bpfolders(os.path.join(path, self.name))
        # Subfolders :
        sfold = ['config', 'database', 'feature', 'classified', 'multifeature',
                 'figure', 'backup', 'anatomy', 'setting', 'other', 'script']
        _ = [self._bpfolders(os.path.join(path, self.name, k)) for k in sfold]  # noqa
        # Add the study to the bpsetting file:
        now = datetime.now()
        now_str = [str(k) for k in (now.month, now.day, now.year, now.hour,
                                    now.minute, now.second)]
        # Fill the config file :
        self.config.update({self.name: {}})
        self['path'] = os.path.join(path, self.name)
        self['created'] = '%s/%s/%s, %s:%s:%s' % tuple(now_str)
        self.path = self['path']
        self.created = self['created']
        # Get path to the bp file :
        bp_path = self._path_bpsettings()
        update_json(bp_path, self.config)
        # Check that the entry is added :
        cfg = load_json(bp_path)
        assert self.name in cfg.keys()
        assert all([k in cfg[self.name].keys() for k in ('path', 'created')])
        logger.info('    %s successfully created' % self.name)

    def delete(self):
        """Delete the current study."""
        logger.warning('Delete the study %s? [y/n]' % self.name)
        user_input = input()
        if user_input is 'y':
            assert os.path.isdir(self.path)
            bp_path = self._path_bpsettings()
            rmtree(self['path'])
            del self.config[self.name]
            save_json(bp_path, self.config)
            logger.info('%s has been deleted.' % self.name)

    # -------------------------------------------------------------
    # Manage Files:
    # -------------------------------------------------------------
    def search(self, *args, folder='', intersection=True, case=True,
               full_path=True, sort=True, exclude=None, split=None,
               verbose=None):
        """Get a list of files.

        Parameters
        ----------
        args : string
            Add some filters to get a restricted list of files,
            according to the defined filters
        folder : string | ''
            Define a folder to search. By default, no folder is specified
            so the search will focused on the root folder.
        intersection : bool | True
            Specify if the intersection should be considered across search
            patterns or the union.
        case : bool | True
            Define if the search method have to take care of the case.
        full_path : bool | True
            Get files with full path (True) or only file names (False).
        sort : bool | True
            Sort files.
        exclude : list | None
            Exclude a list of files.
        split : int | None
            Split the returned list of filst into smaller list.

        Returns
        -------
        files : list
            A list containing the files found in the folder.
        """
        set_log_level(verbose)
        # Get path and files in the folder :
        dir_path = os.path.join(self.path, folder)
        assert os.path.isdir(dir_path)
        def_file = os.listdir(dir_path)
        # Case sentitive (or not) :
        if not case:
            dir_file = [k.lower() for k in def_file]
            args = [k.lower() for k in args]
        else:
            dir_file = def_file

        if not len(args):
            files = dir_file
        else:
            if intersection:
                files = self._search_files(def_file, dir_file, args,
                                           intersection)
            else:
                files = []
                for a in args:
                    files += self._search_files(def_file, dir_file, [a],
                                                intersection)
                sort = False

        # Exclude files :
        if isinstance(exclude, (list, tuple, np.ndarray)):
            files = [k for k in files if k not in exclude]
        logger.info("    %i files found : %s" % (len(files), ', '.join(files)))
        # Full path :
        if full_path:
            files = [os.path.join(dir_path, k) for k in files]
        # Sort :
        if sort:
            files.sort()
        # Split :
        if isinstance(split, int) and (split < len(files)):
            split = len(files) if split == -1 else split
            files = [k.tolist() for k in np.array_split(files, split)]
        return files

    @staticmethod
    def _search_files(def_file, dir_file, args, intersection):
        """Get the list of files according to string patterns."""
        n_files, n_args = len(dir_file), len(args)
        filter_feat = np.zeros((n_files, n_args), dtype=bool)
        for i, k in product(range(n_files), range(n_args)):
            filter_feat[i, k] = bool(dir_file[i].find(args[k]) + 1)
        fcn = np.all if intersection else np.any
        is_searched = fcn(filter_feat, 1)
        return [k for k, i in zip(def_file, is_searched) if i]

    def path_to_folder(self, folder, force=False):
        """Get the path to a folder.

        Parameters
        ----------
        folder : string
            Name of the folder.
        force : bool | False
            Force the creation if doesn't exist.
        """
        path = os.path.join(self.path, folder)
        if not os.path.isdir(path) and force:
            self._bpfolders(path)
        return path

    def add_folder(self, name):
        """Add a folder to the study.

        Parameters
        ----------
        name : string
            Name of the folder
        """
        full_path = os.path.join(self.path, name)
        if os.path.isdir(full_path):
            logger.warning("Folder %s already exist" % name)
            return full_path
        self._bpfolders(full_path)
        assert os.path.isdir(full_path)
        logger.info("    Folder %s added" % name)
        return full_path

    def load(self, file, folder=None, verbose=None):
        """Load a file.

        This method support to load :

            * .mat (Matlab)
            * .pickle
            * .npy and .npz
            * .json
            * .txt

        Parameter
        ---------
        file : string
            Name of the file.
        folder : string | None
            Specify where the file is located. If `folder` is None, full path
            should be given.

        Returns
        -------
        file
            The loaded file.
        """
        set_log_level(verbose)
        folder = '' if not isinstance(folder, str) else folder
        full_path = os.path.join(self.path, folder, file)
        arch = load_file(full_path)
        logger.info('    %s loaded' % file)
        return arch

    def save(self, file, *arg, folder=None, compress=False, **kwargs):
        """Save a file.

        This method support to save :

            * .mat (Matlab)
            * .pickle
            * .npy and .npz
            * .json

        Parameters
        ----------
        file : string
            Name of the file
        folder : string | None
            Specify where the file need to be saved. If `folder` is None, full
            path should be given.
        args : tuple
            Additional arguments for saving .npy arrays
        kwargs : dict | {}
            Additional arguments for saving .mat, .pickle, .npz and .json files
        """
        folder = '' if not isinstance(folder, str) else folder
        full_path = os.path.join(self.path, folder, file)
        save_file(full_path, *arg, compress=compress, **kwargs)
        logger.info("    %s saved" % full_path)

    def load_config(self, file, entry=None):
        """Load a configuration file.

        Parameters
        ----------
        file : string
            Name of the JSON config file.
        entry : str | None
            Entry in the config file.
        """
        assert '.json' in file
        cfg = self.load(file, folder='config')
        if isinstance(entry, str) and (entry in cfg.keys()):
            return cfg[entry]
        else:
            return cfg

    def save_config(self, file, kw):
        """Save a configuration file.

        Parameters
        ----------
        file : string
            Name of the JSON config file.
        kw : dict
            Dictionary to save.
        """
        assert '.json' in file
        self.save(file, folder='config', **kw)

    def update_config(self, file, kw, backup=True):
        """Update a configuration file.

        Parameters
        ----------
        file : string
            Name of the JSON config file.
        kw : dict
            Dictionary to use for the update.
        backup : bool | True
            Create a backup of the configuration to /study/backup/
        """
        assert '.json' in file
        backup = self.path_to_folder('backup') if backup else None
        full_path = os.path.join(self.path, 'config', file)
        update_json(full_path, kw, backup)
        logger.info("    %s configuration file has been updated" % file)

    def load_script(self, filename):
        """Load a script.

        Parameters
        ----------
        filename : string
            Name of the .py file to load.

        Returns
        -------
        mod : module
            The desired module to load.
        """
        from importlib.util import spec_from_file_location, module_from_spec
        full_path = os.path.join(self.path, 'script', filename)
        spec = spec_from_file_location('module_name', full_path)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @property
    def studies(self):
        """Get doc."""
        bp_path = self._path_bpsettings()
        return load_json(bp_path).keys()

    def _bpfolders(self, directory):
        """Check if a folder exist otherwise, create it."""
        if not os.path.exists(directory):
            os.makedirs(directory)

    def _load_bpsettings(self):
        """Load the bpsettings file."""
        return load_json(self._path_bpsettings())

    def _path_bpsettings(self):
        """Get the path of bpsettings."""
        dir_path = os.path.dirname(os.path.realpath(__file__))
        bp_path = re.findall('(.*?)brainpipe', dir_path)[0]
        return os.path.join(*(bp_path, 'brainpipe', BP_FILE))


class figure(object):
    """Manage figure
    """

    def __init__(self, path):
        self._path = path

    def save(self, name, dpi=None, gcf=None, bbox_inches='tight', **kwargs):
        fname = os.path.join(self._path, name)
        if not gcf:
            plt.savefig(safety_save(fname), dpi=dpi, bbox_inches=bbox_inches)
        if gcf:
            gcf.savefig(safety_save(fname), dpi=dpi, bbox_inches=bbox_inches)

    def load(self, name):
        return plt.imread(os.path.join(self._path, name))

    def show(self, name, **kwargs):
        im = self.load(name)
        plt.box('off')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(im, **kwargs)
        plt.show()

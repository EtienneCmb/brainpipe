from os import listdir
from os.path import splitext, join
from datetime import datetime

from shutil import rmtree
from scipy.io import loadmat
import pickle
import matplotlib.pyplot as plt

import brainpipe
from brainpipe.sys.tools import loadfile, savefile, _safetySave
from brainpipe.sys.utils._bpstudy import *

import numpy as n

__all__ = ['study']


class study(object):
    """Create and manage a study with a files database.

    Args:
        name: string, optional [def : None]
            Name of the study. If this study already exists, this will load
            the path with the associated database.

    Example:
        >>> # Define variables:
        >>> path = '/home/Documents/database'
        >>> studyName = 'MyStudy'

        >>> # Create a study object :
        >>> studObj = study(name=studyName)
        >>> studObj.add(path)   # Create the study
        >>> studObj.studies()   # Print the list of studies

        >>> # Manage files in your study :
        >>> fileList = studObj.search('filter1', 'filter2', folder='features')
        >>> # Let say that fileList contain two files : ['file1.mat', 'file2.pickle']
        >>> # Load the second file :
        >>> data = studObj.load('features', 'file2.pickle')
    """

    def __init__(self, name=None):
        if name:
            self.name = name
            _check_bpsettings_exist()
            _check_study_exist(self)
            _update_bpsettings()
            try:
                self.dataset = dataset(join(self.path, 'dataset'))
                self.feature = feature(join(self.path, 'feature'))
                self.figure = figure(join(self.path, 'figure'))
            except:
                print(name+' not found. use self.add() to add it')

    def __str__(self):
        return 'Study name: '+self.name+', path = '

    # -------------------------------------------------------------
    # Manage Files:
    # -------------------------------------------------------------
    def add(self, path):
        """Add a new study

        Args:
            path: string
                path to your study.

            The following folders are going to be created :
                - name: the root folder of the study. Same name as the study
                - /database: datasets of the study
                - /feature: features extracted from the diffrents datasets
                - /classified: classified features
                - /multifeature: multifeatures files
                - /figure: figures of the study
                - /physiology: physiological informations
                - /backup: backup files
                - /setting: study settings
                - /other: any other kind of files
        """
        # Main studyName directory :
        _bpfolders(join(path, self.name))
        # Subfolders :
        _bpfolders(join(path, self.name, 'database'))
        _bpfolders(join(path, self.name, 'feature'))
        _bpfolders(join(path, self.name, 'classified'))
        _bpfolders(join(path, self.name, 'multifeature'))
        _bpfolders(join(path, self.name, 'figure'))
        _bpfolders(join(path, self.name, 'backup'))
        _bpfolders(join(path, self.name, 'physiology'))
        _bpfolders(join(path, self.name, 'setting'))
        _bpfolders(join(path, self.name, 'other'))
        # Add the study to the bpsetting file:
        now = datetime.now()
        creation = (str(now.month)+'/'+str(now.day)+'/'+str(now.year),
                    str(now.hour)+':'+str(now.minute)+':'+str(now.second))
        _add_bpsettings_entry(self.name, path, creation)
        _update_bpsettings()
        print(self.name+' has been successfully created')
        _check_study_exist(self)

    def delete(self):
        """Delete the current study
        """
        print('Delete the study '+self.name+'? [y/n]')
        userInput = input()
        if userInput is 'y':
            try:
                rmtree(self.path)
                print(self.name+' has been deleted')
            except:
                print('No folder found')
            _update_bpsettings()

    # -------------------------------------------------------------
    # Manage Files:
    # -------------------------------------------------------------
    def search(self, *args, folder='', lower=True):
        """Get a list of files

        Args:
            args: string, optional
                Add some filters to get a restricted list of files,
                according to the defined filters

            folder: string, optional [def: '']
                Define a folder to search. By default, no folder is specified
                so the search will focused on the root folder.

            lower: bool, optional [def: True]
                Define if the search method have to take care of the case. Use
                False if case is important for searching.

        Return:
            A list containing the files found in the folder.
        """

        ListFeat = listdir(join(self.path, folder))

        if args == ():
            return ListFeat
        else:
            filterFeat = n.zeros((len(args), len(ListFeat)))
            for k in range(len(args)):
                for i in range(len(ListFeat)):
                    # Case of lower case :
                    if lower:
                        strCmp = ListFeat[i].lower().find(
                            args[k].lower()) != -1
                    else:
                        strCmp = ListFeat[i].find(args[k]) != -1
                    if strCmp:
                        filterFeat[k, i] = 1
                    else:
                        filterFeat[k, i] = 0
        return [ListFeat[k] for k in n.where(n.sum(filterFeat, 0) == len(
                    args))[0]]

    def load(self, folder, name):
        """Load a file. The file can be a .pickle or .mat

        Args:
            folder: string
                Specify where the file is located

            file: string
                the complete name of the file

        Return:
            A dictionary containing all the variables.
        """
        return loadfile(join(self.path, folder, name))

    def save(self, folder, name, *arg, **kwargs):
        savefile(join(self.path, folder, name), *arg, **kwargs)

    # -------------------------------------------------------------
    # Static methods :
    # -------------------------------------------------------------
    @staticmethod
    def studies():
        """Get the list of all defined studies
        """
        bpCfg = _path_bpsettings()
        with open(bpCfg, "rb") as f:
            bpsettings = pickle.load(f)
        _update_bpsettings()
        print(bpsettings)

    @staticmethod
    def update():
        """Update the list of studies
        """
        _update_bpsettings()


class dataset(object):
    """Manage dataset
    """
    def __init__(self, path):
        pass

    def save(self):
        pass

    def load(self):
        pass


class feature(object):
    """Manage feature
    """
    def __init__(self, path):
        self._path = path

    def save(self):
        pass

    def load(self):
        pass


class figure(object):
    """Manage figure
    """
    def __init__(self, path):
        self._path = path

    def save(self, name, dpi=None, gcf=None, bbox_inches='tight', **kwargs):
        fname = join(self._path, name)
        if not gcf:
            plt.savefig(_safetySave(fname), dpi=dpi, bbox_inches=bbox_inches)
        if gcf:
            gcf.savefig(_safetySave(fname), dpi=dpi, bbox_inches=bbox_inches)
        print('Saved to '+fname)

    def load(self, name):
        return plt.imread(join(self._path, name))

    def show(self, name, **kwargs):
        im = self.load(name)
        plt.box('off')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(im, **kwargs)
        plt.show()

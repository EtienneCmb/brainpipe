import numpy as n
from os.path import abspath, dirname, join
from inspect import getfile, currentframe
from scipy.io import loadmat
from pandas import DataFrame, concat
import pandas as pd
import pickle
pd.options.mode.chained_assignment = None


def pos2label(posC, mask, hdr, gray, label, r=5, nearest=True):
    """From the xyz coordonates, get the labels of gyrus, hemisphere...
    """
    pos = n.ndarray.tolist(posC)
    if type(pos[0]) is not list:
        pos = [pos]
    pos = mni2tal(pos)
    ind = [coord2ind(k, mask, hdr, gray, r=r, nearest=nearest)
           for k in pos]
    sub = label.iloc[ind]
    sub['X'], sub['Y'], sub['Z'] = posC[:, 0], posC[:, 1], posC[:, 2]
    sub = sub.set_index([list(n.arange(sub.shape[0]))])
    return sub


def coord2ind(pos, mask, hdr, gray, r=5, nearest=True):
    """Find the correspondance between coordonate and atlas index
    """
    # Apply a transformation of position:
    pos.extend([1])
    sub = list(
        n.around(n.array(n.linalg.lstsq(hdr, n.matrix(pos).T)[0].T)[0]).astype(
            int))
    # Find the index with the nearest option:
    if nearest:
        if sub[2] > gray.shape[2]:
            sub[2] = gray.shape[2]-1
        tranche = n.squeeze(gray[:, :, sub[2]])
        # Euclidian distance :
        dist = 100*n.ones((tranche.shape))
        u, v = n.where(tranche == 1)
        for k in range(0, len(u)):
            dist[u[k], v[k]] = n.math.sqrt((sub[1]-v[k])**2 + (sub[0]-u[k])**2)
        mindist = dist.min()
        umin, vmin = n.where(dist == mindist)
        if mindist < r:
            ind = mask[umin[0], vmin[0], sub[2]]-1
        else:
            ind = -1
    else:
        try:
            ind = mask[sub[0], sub[1], sub[2]]-1
        except:
            ind = -2
    return ind


def loadatlas(atlas='tal', r=5):
    """Load the atlas from the brainpipe module
    """
    B3Dpath = dirname(
        abspath(join(getfile(currentframe()), '..', '..', 'atlas')))
    # Load AAL atlas :
    if atlas is 'AAL':  # PAS FINIT
        AAL = scio.loadmat(B3Dpath + '/Atlas/Labels/BrainNet_AAL_Label')
        hdr, label, mask = AAL['AAL_hdr'], AAL['AAL_label'], AAL['AAL_vol']

    # Load talairach atlas :
    if atlas is 'tal':
        with open(B3Dpath + '/atlas/labels/talairach_atlas.pickle', "rb") as f:
            TAL = pickle.load(f)
        label = TAL['label']
        strGM = ['No Gray Matter found within +/-'+str(r)+'mm']
        label = concat([label, DataFrame({'hemisphere': [strGM], 'lobe':[
                       strGM], 'gyrus':[strGM], 'matter':[strGM], 'brodmann':[
            0]})])
        label = label.set_index([list(n.arange(label.shape[0]))])
        return TAL['hdr'], TAL['mask'], TAL['gray'], label


def mni2tal(posmni):
    """Transform coordonates from mni to talairach
    """
    upT = spm_matrix([0, 0, 0, 0.05, 0, 0, 0.99, 0.97, 0.92])
    downT = spm_matrix([0, 0, 0, 0.05, 0, 0, 0.99, 0.97, 0.84])
    pos = []
    for k in posmni:
        tmp = k[-1] < 0
        k.extend([1])
        k = n.matrix(k).T
        if tmp:
            k = downT * k
        else:
            k = upT * k
        pos.append(list(n.array(k.T)[0][0:3]))
    return pos


def spm_matrix(P):
    """Matrix transformation
    """

    q = [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    P.extend(q[len(P):12])

    T = n.matrix(
        [[1, 0, 0, P[0]], [0, 1, 0, P[1]], [0, 0, 1, P[2]], [0, 0, 0, 1]])
    R1 = n.matrix([[1, 0, 0, 0], [0, n.cos(P[3]), n.sin(P[3]), 0],
                   [0, -n.sin(P[3]), n.cos(P[3]), 0], [0, 0, 0, 1]])
    R2 = n.matrix([[n.cos(P[4]), 0, n.sin(P[4]), 0], [0, 1, 0, 0],
                   [-n.sin([P[4]]), 0, n.cos(P[4]), 0], [0, 0, 0, 1]])
    R3 = n.matrix([[n.cos(P[5]), n.sin(P[5]), 0, 0], [-n.sin(P[5]), n.cos(
                   P[5]), 0, 0],
                   [0, 0, 1, 0], [0, 0, 0, 1]])
    Z = n.matrix([[P[6], 0, 0, 0], [0, P[7], 0, 0],
                  [0, 0, P[8], 0], [0, 0, 0, 1]])
    S = n.matrix([[1, P[9], P[10], 0], [0, 1, P[11], 0],
                  [0, 0, 1, 0], [0, 0, 0, 1]])
    return T*R1*R2*R3*Z*S

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:08:17 2024

@author: KOlson
"""
# External imports
import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
import pytdms
import os.path as osp
from typing import Literal
import pathlib

# Internal imports
try:
    from helpers import are_equal
except ModuleNotFoundError:
    from MetroLaserLARS.helpers import are_equal


class LarsData:
    """Class for Lars Data. Includes name, path, time, pztV, ldvV, freq, and vel arrays."""
    name: str
    path: str
    time: NDArray
    pztV: NDArray
    ldvV: NDArray
    freq: NDArray
    vel: NDArray
    newvel: NDArray
    peaks: dict
    analyzed_this_session: bool

    def __init__(self, name: str = None, path: str = None, time: NDArray = None, pztV: NDArray = None,
                 ldvV: NDArray = None, freq: NDArray = None, vel: NDArray = None):
        self.name = name
        self.path = path
        self.time = time
        self.pztV = pztV
        self.ldvV = ldvV
        self.freq = freq
        self.vel = vel
        self.newvel = np.zeros_like(vel)
        self.peaks = {}
        return

    def __eq__(self, other):
        if isinstance(other, LarsData):
            return are_equal(vars(self), vars(other))
        return False

    @classmethod
    def from_file(cls, path: str, permanent_path: Literal[None, str] = None, **settings):
        """
        Loads LARS data from `path`. Expects files with 5 columns, corresponding to:
            0: time
            1: piezoelectric transducer voltage
            2: laser doppler vibrometer voltage
            3: frequency
            4: velocity

        Parameters
        ----------
        path : str
            path to data to load. May point to temporary files or files in a temporary directory for loading speed.
        permanent_path : Literal[None,str], optional
            If supplied, the `name` and `path` attributes are derived from `permanent_path`.
            Should be supplied if `path` is a temporary file or directory because the name is more likely to be
            meaningful and because the temporary file or folder may be cleaned up before the LarsData object.
            The default is None, and uses `path`.

        Returns
        -------
        LarsData
            LarsData class object with data from `path`.

        """

        new_data_format = settings['new_data_format'] if 'new_data_format' in settings else 'none'
        permanent_path = path if permanent_path is None else permanent_path

        nn = {'f': 'Frequency (Hz)', 'a': 'FT Amplitude (um/s)', 'v': 'LDV Amplitude (um/s)',
              'p': 'PZT Amplitude (V)', 'R': 'Sample Rate (Hz)', 'T': 'Sampling Duration (s)', 't': 'Time (s)'}

        path_no_ext, ext = osp.splitext(path)
        if ext == '.npz':
            data = dict(np.load(path))
        elif ext == '.tdms':
            data = {}
            _, rawdata = pytdms.read(path)
            for k, v in rawdata.items():
                k = k.split(b"'")[1].decode()+' '+k.split(b"'")[3].decode() if k.split(b"'")[3] != b"Untitled" else k.split(b"'")[1].decode()
                v = np.array(v)  # if len(v) > 1 else v[0]
                if nn['v'] in k:
                    if nn['v'] not in data:
                        data[nn['v']] = v
                    else:
                        data[nn['v']] = np.vstack((data[nn['v']], v))
                else:
                    data[k] = v
        elif ext == '.all':
            data = np.loadtxt(path)
            return cls(name=osp.basename(permanent_path), path=permanent_path, time=data[:, 0], pztV=data[:, 1],
                       ldvV=data[:, 2], freq=data[:, 3], vel=data[:, 4])
        elif ext == '.csv':
            df = pd.read_csv(path)
            data = dict(zip(df.columns.values, df.to_numpy().T))
            for k, v in list(data.items()):
                data[k] = v[~np.isnan(v)]
                if nn['v'] in k:
                    if nn['v'] not in data:
                        data[nn['v']] = data.pop(k)
                    else:
                        data[nn['v']] = np.vstack((data[nn['v']], data.pop(k)))

        else:
            raise f"""Tried to load LARS data form an {ext} file, which is an invalid file type.
Only load .npz, .tdms, .all, or .csv files. Full path: {permanent_path}"""
        if ext != '.all':
            if new_data_format in ['.npz', 'both']:
                np.savez_compressed(path_no_ext+'.npz', **data)
            if new_data_format in ['.csv', 'both']:
                dataout = data.copy()
                pop = False
                for k, v in list(dataout.items()):
                    if nn['v'] in k and data[nn['v']].ndim > 1:
                        pop = True
                        for i, row in enumerate(v):
                            dataout[nn['v']+' '+str(i)] = row
                if pop:
                    dataout.pop(nn['v'])
                df = pd.DataFrame.from_dict(dataout, orient='index').transpose()
                df.to_csv(path_no_ext+'.csv', index=False)

            if data[nn['v']].ndim > 1:
                ldvV = np.mean(data[nn['v']], axis=0)
            else:
                ldvV = data[nn['v']]
            return cls(name=osp.basename(permanent_path), path=permanent_path, time=data[nn['t']], pztV=data[nn['p']],
                       ldvV=ldvV, freq=data[nn['f']], vel=data[nn['a']])

    @classmethod
    def from_subdata(cls, c):
        return cls(name=pathlib.Path(c.path).parts[-2], path=osp.split(c.path)[0], time=c.time, pztV=c.pztV,
                   ldvV=c.ldvV, freq=c.freq, vel=c.vel)


def all_equal(l: list[ArrayLike]) -> bool:
    """
    Checks if all arrays in a list are equal to each other.

    Parameters
    ----------
    l : list[ArrayLike]
        List of arrays to check.

    Returns
    -------
    bool
        Whether all arrays in `l` are equal.

    """
    for el in l[1:]:
        if not (el == l[0]).all():
            return False
    return True


def combine(alldata: list[LarsData], combine: Literal['max', 'mean'] = 'max') -> LarsData:
    """
    Generates a combined LarsData class from a list of LarsData classes.
    `combine` defines how the velocity data should be combined.

    Parameters
    ----------
    alldata : list[LarsData]
        List of LarsData objects to combine.
    combine : Literal['max','mean'], optional
        Combination method. At each frequency, the maximum (`'max'`) or average (`'mean'`) velocity
        is used for the combined data. The default is 'max'.

    Returns
    -------
    LarsData
        A LarsData object with combined data.

    """
    if not all_equal([data.freq for data in alldata]):
        print('Warning: frequencies of individual measurements are not all the same, combining them is dangerous')

    combined_data = LarsData.from_subdata(alldata[0])

    if combine == 'mean':
        combined_data.vel = np.mean([data.vel for data in alldata], axis=0)
    elif combine == 'max':
        combined_data.vel = np.maximum.reduce([data.vel for data in alldata])
    else:  # default to max
        combined_data.vel = np.maximum.reduce([data.vel for data in alldata])

    return combined_data

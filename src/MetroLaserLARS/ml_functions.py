# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:29:15 2024

@author: KOlson
"""
# External imports
import numpy as np
import scipy
import pathlib
from numpy.typing import NDArray

# Internal imports
try:
    from LarsDataClass import LarsData
    from filters import sgf
    from cnn.models import ConvNet
    from cnn.preprocessing import LabelEncoder
    from app_helpers import resource_path
    from helpers import peaks_dict_from_array
except ModuleNotFoundError:
    from MetroLaserLARS.LarsDataClass import LarsData  # type: ignore
    from MetroLaserLARS.filters import sgf  # type: ignore
    from MetroLaserLARS.cnn.models import ConvNet  # type: ignore
    from MetroLaserLARS.cnn.preprocessing import LabelEncoder  # type: ignore
    from MetroLaserLARS.app_helpers import resource_path  # type: ignore
    from MetroLaserLARS.helpers import peaks_dict_from_array  # type: ignore

default_ml_weights_path = 'output/weights/009.weights.h5'


def load_model(**settings):
    ml_classes = 3 if 'ml_classes' not in settings else settings['ml_classes']
    ml_windows = 256 if 'ml_windows' not in settings else settings['ml_windows']
    ml_weights_path = default_ml_weights_path if 'ml_weights_path' not in settings else settings['ml_weights_path']
    ml_input_size = 8192 if 'ml_input_size' not in settings else settings['ml_input_size']

    if ml_weights_path == 'default weights':
        ml_weights_path = default_ml_weights_path

    label_encoder = LabelEncoder(ml_windows)
    model = ConvNet(
        filters=[64, 128, 128, 256, 256],
        kernel_sizes=[9, 9, 9, 9, 9],
        dropout=0.0,
        pool_type='max',
        pool_sizes=[2, 2, 2, 2, 2],
        conv_block_size=1,
        input_shape=(ml_input_size, 1),
        output_shape=(ml_windows, ml_classes),
        residual=False
    )
    if ml_weights_path != default_ml_weights_path:
        try:
            model.load_weights(ml_weights_path)
        except Exception:
            print('Loading custom ML peak finding weights failed, reverting to default weights')
            model.load_weights(resource_path(default_ml_weights_path))
    else:
        model.load_weights(resource_path(ml_weights_path))
    return model, label_encoder


def analyze_data(data: LarsData, **settings) -> tuple[dict, NDArray, NDArray, NDArray, str]:
    """
    Smooths raw LARS data with `filters.sgf()`,
    and fits peaks with a pre-trained convolutional neural network.

    For `sgf` information, see https://medium.com/pythoneers/introduction-to-the-savitzky-golay-filter-a-comprehensive-guide-using-python-b2dd07a8e2ce.

    Parameters
    ----------
    data : LarsData
        LarsData to be analyzed.
    settings : optional kwargs
        See `main()`.

    Returns
    -------
    peaks : dict
        A dictionary of peak data, containing the keys:
            'count': Number of peaks
            'indices': Indices of peaks
            'positions': The locations of peaks in `x`
            'heights': The heights of peaks in `y` units
            'widths': The widths of peaks in `x` units. Assumes `x` is evenly spaced.
            'lefts', 'rights': The left and right edges of peaks in `x` units. Assumes `x` is evenly spaced.
    freqs : NDArray
        Frequencies corresponding to the velocities.
    vels : NDArray
        Velocities before smoothing and baseline correction.
    newvels : NDArray
        Velocities after smoothing and baseline correction.
    name: str
        Name of the data.

    """
    slc_limits = settings['slc_limits'] if 'slc_limits' in settings else (12000, 60000)
    sgf_applications = settings['sgf_applications'] if 'sgf_applications' in settings else 2
    sgf_windowsize = settings['sgf_windowsize'] if 'sgf_windowsize' in settings else 101
    sgf_polyorder = settings['sgf_polyorder'] if 'sgf_polyorder' in settings else 0
    PRINT_MODE = settings['PRINT_MODE'] if 'PRINT_MODE' in settings else 'none'

    freqs = data.freq
    slc = np.logical_and(freqs > slc_limits[0], freqs < slc_limits[1])
    folder = pathlib.Path(data.path).parts[-2]
    freqs = freqs[slc]
    name = data.name

    if len(data.freq) > 0 and len(data.vel) == 0:  # has freqs but not vels, it is a simulation file with the peaks listed in freqs
        peaks = peaks_dict_from_array(freqs)
        if PRINT_MODE == 'full':
            peaklist = peaks['positions'][np.logical_and(peaks['positions'] > slc_limits[0],
                                                         peaks['positions'] < slc_limits[1])]
            print(f"""For {folder}/{name}:
              simulated peaks at:     {peaklist/1000} kHz""")
        return peaks, freqs, data.vel, data.vel, data.name

    vels = data.vel[slc]

    newvels = sgf(vels, n=sgf_applications, w=sgf_windowsize, p=sgf_polyorder)

    probs, locs, areas = predict(freqs, newvels, settings['model'], settings['label_encoder'], settings)

    peaks = peaks_dict_from_array(locs)
    return peaks, freqs, vels, newvels, name


def predict(x: NDArray, y: NDArray, model, label_encoder, settings):
    ml_peak_fit_threshold = 0.01 if 'ml_threshold' not in settings else settings['ml_threshold']
    ml_input_size = 8192 if 'ml_input_size' not in settings else settings['ml_input_size']

    x_prep = np.linspace(np.min(x), np.max(x), ml_input_size)
    y_prep = scipy.interpolate.interp1d(x, y)(x_prep)
    y_norm = y_prep.copy()[None, :, None]/np.max(y_prep)
    preds = model(y_norm)[0]
    probs, locs, areas = label_encoder.decode(preds, threshold=ml_peak_fit_threshold)
    locs *= np.max(x)-np.min(x)
    locs += np.min(x)
    return probs, locs, areas

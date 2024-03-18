import copy
import logging
import os

import mne
from mne import Annotations
import pyxdf
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
from tqdm import tqdm
from autoreject import get_rejection_threshold
from pyprep import NoisyChannels
import statsmodels.api as sm
import numpy as np
import pandas as pd

from utils.file_mgt import *


"""Loads and clean the EEG data.
Features are extracted and saved in a separate file.
"""


def get_raw_from_xdf(xdf_file_path: str, ref_electrode: str = "") -> mne.io.Raw:
    """This function loads an XDF EEG file, and returns the corresponding mne.io.Raw object.

    Parameters
    ----------
    xdf_file_path : str
        Path to the XDF file.
    ref_electrode : str
        If not empty, a referential montage with that electrode is used, otherwise an average montage is used.
    """
    streams, _ = pyxdf.load_xdf(xdf_file_path)

    # Find where the EEG data is located within the data structure
    assert len(streams) == 2, (
        "Unexpected XDF data structure : expecting 2 streams, got " + str(len(streams))
    )
    if streams[1]["time_series"].shape[0] > streams[0]["time_series"].shape[0]:
        stream_index = 1
        stream_index_markers = 0
    else:
        stream_index = 0
        stream_index_markers = 1

    # Count EEG channels and find the reference channel's index
    channels_info = streams[stream_index]["info"]["desc"][0]["channels"][0]["channel"]
    eeg_channel_count = 0
    ref_channel = -1
    for index, e in enumerate(channels_info):
        if e["type"][0] == "EEG":
            eeg_channel_count += 1
        if e["label"][0] == ref_electrode:
            ref_channel = index

    # Extract channels' info
    data = streams[stream_index]["time_series"].T
    # It is assumed that the EEG channels are the first ones
    data = data[:eeg_channel_count]
    # micro V to V and preamp gain ???
    data[:] *= 1e-6  # / 2
    sfreq = float(streams[stream_index]["info"]["nominal_srate"][0])
    channel_names = [
        e["label"][0]
        + (
            (" - " + ref_electrode)
            if (e["label"][0] != ref_electrode) and ref_electrode != ""
            else ""
        )
        for e in channels_info[:eeg_channel_count]
    ]

    # Data format check
    assert eeg_channel_count > 0, "No EEG channels were found."
    if ref_electrode != "":
        assert ref_channel > -1, "The specified reference electrode was not found."
    for e in channel_names:
        assert e[0] in ["F", "C", "T", "P", "O"], "The channel names are unexpected."
    assert sfreq > 0.0, "The sampling frequency is not a positive number."

    # Create the men.io.Raw object
    info = mne.create_info(channel_names, sfreq, ["eeg"] * eeg_channel_count)
    raw = mne.io.RawArray(data, info, verbose=False)

    # Event annotations
    origin_time = streams[stream_index]["time_stamps"][0]
    markers_time_stamps = [
        e - origin_time for e in streams[stream_index_markers]["time_stamps"]
    ]
    markers_nb = len(markers_time_stamps)
    markers = Annotations(
        onset=markers_time_stamps,
        duration=[10] * 3 + [25] * 5 + [25] * 5,
        description=["Audio"] * 3
        + ["Mental arithmetics moderate"] * 5
        + ["Mental arithmetics hard"] * 5,
        ch_names=[channel_names] * markers_nb,
    )
    raw.set_annotations(markers)

    # Set the reference montage
    if ref_electrode != "":
        raw = raw.set_eeg_reference(ref_channels=[ref_electrode], verbose=False)
    else:
        raw = raw.set_eeg_reference(verbose=False)  # Use the average montage

    # Set the electrode positions
    channel_mapping = {"Fp1":"AFp1", "Fz": "AFF1h", "F3": "AF7", "F7": "AFF5h", "F9":"FT7", "FC5": "FC5", "FC1":"FC3", "C3":"FCC3h",
        "T7":"FFC1h", "CP5": "FCC1h", "CP1": "CCP3h", "Pz":"CCP1h", "P3":"CP1", "P7": "CP3", "P9":"CPP3h", "O1":"P1",
        "Oz":"AFp2", "O2":"AFF2h", "P10":"AF8", "P8":"AFF6h", "P4":"FT8", "CP2":"FC6", "CP6":"FC4", "T8":"FCC4h",
        "C4":"FFC2h", "Cz":"FCC2h", "FC2":"CCP4h", "FC6": "CCP2h", "F10":"CP2", "F8":"CP4", "F4":"CPP4h", "Fp2":"P2"}
    raw.rename_channels(channel_mapping)
    cap_montage = mne.channels.make_standard_montage("brainproducts-RNP-BA-128")
    raw.set_montage(cap_montage)

    return raw


def filter_raw(raw: mne.io.Raw) -> mne.io.Raw:
    """Filters a mne.io.Raw object : bandpass filter between 1 and 70 Hz, and notch filter at 50 Hz and some harmonics."""
    raw_copy = copy.deepcopy(raw)
    raw_copy.filter(l_freq=1, h_freq=70, verbose=False)
    raw_copy.notch_filter(np.arange(50, 250, 50), verbose=False)
    return raw_copy


def epochs_from_raw(raw: mne.io.Raw) -> mne.Epochs:
    """
    Returns a mne.Epochs object created from an annotated mne.io.Raw object.
    The length of the epochs is arbitrarily set, based on how the data was acquired.
    """
    events, events_id = mne.events_from_annotations(raw)
    return mne.Epochs(
        raw, events, event_id=events_id, preload=True, tmin=0, tmax=25, baseline=None
    )


def add_brain_wave_types_lines_on_pyplot_figure():
    """To be called after : fig = spectrum.plot()"""
    plt.axvline(x=0, color="b")
    # Delta
    plt.axvline(x=4, color="b")
    # Theta
    plt.axvline(x=8, color="b")
    # Alpha
    plt.axvline(x=13, color="b")
    # Beta
    plt.axvline(x=30, color="b")
    # Gamma


def main():
    
    seed = np.random.random()
    logging.info("Random seed is {}".format(seed))

    mne.set_config("MNE_BROWSER_BACKEND", "qt")
    mne.set_log_level("WARNING")

    paths = list()
    paths = get_random_xdf_file_paths(10, seed)
    # paths = get_random_xdf_file_paths_one_session(seed)

    feature_set = {} # Key: Recording ID - Value: feature vector

    for path in tqdm(paths):

        logging.info("Now working with file {}".format(path))

        # ------ Part 1 : Raw + Bad channel detection and handling + Filter ---------------------------

        try:
            raw = get_raw_from_xdf(path).load_data()
        except Exception as e:
            print(e)
            continue

        handler = NoisyChannels(raw)
        handler.find_bad_by_deviation()  # Detect channels with abnormally high or low overall amplitudes.
        handler.find_bad_by_hfnoise()  # Detect channels with abnormally high amounts of high-frequency noise.
        # handler.find_bad_by_correlation() # Detect channels that sometimes don’t correlate with any other channels
        bad_channels = handler.get_bads()
        logging.info("Bad channels found by pyprep ({}) : {}".format(len(bad_channels), bad_channels))
        raw.info["bads"] = bad_channels
        if len(bad_channels) > 0 : raw = raw.interpolate_bads()
        raw = raw.set_eeg_reference(ref_channels="average")

        raw = filter_raw(raw)

        # ------ Part 2 : Epoched (split) + Autoreject to drop bad epochs -----------------------------

        epochs = epochs_from_raw(raw).load_data() # TODO : consider the decim parameter to downsample and save memory
        del raw

        reject = get_rejection_threshold(epochs, verbose=False) # look at AutoRejct.fit_transform for interpolation
        logging.info("The rejection dictionary is {}".format(reject))
        n = len(epochs.selection)
        epochs.drop_bad(reject=reject)
        logging.info("{} epochs were dropped by Autoreject".format(n - len(epochs.selection)))
        if set(epochs.picks).intersection([0, 1, 2]) == {}:
            logging.warning("All audio epochs were dropped")
        if set(epochs.picks).intersection([3, 4, 5, 6, 7]) == {}:
            logging.warning("All moderate mental arithmetics epochs were dropped")
        if set(epochs.picks).intersection([8, 9, 10, 11, 12]) == {}:
            logging.warning("All hard mental arithmetics epochs were dropped")

        # ------ Part 3 : Evoked (averaged) -----------------------------------------------------------

        evoked_audio = epochs["Audio"].average()
        evoked_audio.crop(tmin=0, tmax=10)
        # evoked_audio.plot(window_title="Evoked Audio", show=False)
        evoked_maths_1 = epochs["Mental arithmetics moderate"].average()
        evoked_maths_1.crop(tmin=0, tmax=25)
        # evoked_maths_1.plot(window_title="Evoked Mental Arithmetics Moderate", show=False)
        evoked_maths_2 = epochs["Mental arithmetics hard"].average()
        evoked_maths_2.crop(tmin=0, tmax=25)
        # evoked_maths_2.plot(window_title="Evoked Mental Arithmetics Hard", show=False)
        del epochs

        # ------ Part 4 : Feature extraction ----------------------------------------------------------

        # spectrum_audio = evoked_audio.compute_psd(fmax=60)
        # fig = spectrum_audio.plot(average=True)
        # add_brain_wave_types_lines_on_pyplot_figure()
        # spectrum_maths_1 = evoked_maths_1.compute_psd(fmax=60)
        # fig = spectrum_maths_1.plot(average=True)
        # add_brain_wave_types_lines_on_pyplot_figure()
        # spectrum_maths_2 = evoked_maths_2.compute_psd(fmax=60)
        # fig = spectrum_maths_2.plot(average=True)
        # add_brain_wave_types_lines_on_pyplot_figure()
        # plt.show()

        features = []

        # AR process coefficients
        for channel_index in range(evoked_audio.data.shape[0]):
            data = evoked_audio.data[channel_index]
            ar_coefficients, _ = sm.regression.yule_walker(data, order=10, method="mle") # TODO : hyperparameter (AR model order)
            features.extend(ar_coefficients[:5]) # TODO : hyperparameter (number of coefficients to keep)

        feature_set[str(path)] = features
    
    df = pd.DataFrame.from_dict(feature_set, orient="index")
    df.to_csv(os.path.join("data", "processed", "eeg_features.csv"))


if __name__ == "__main__":
    main()


# # ------------------- ICA --------------------
# ica = ICA(max_iter="auto", random_state=2000)#n_components=5,
# ica.fit(raw)
# explained_var_ratio = ica.get_explained_variance_ratio(raw)
# for channel_type, ratio in explained_var_ratio.items():
#     print(
#         f"Fraction of {channel_type} variance explained by all components: " f"{ratio}"
#     )
# ica.plot_sources(raw, show_scrollbars=False)
# # ica.plot_overlay(raw, exclude=[0], picks="eeg")
# ica.exclude = [0]  # indices chosen based on various plots above
# # ica.apply() changes the Raw object in-place, so let's make a copy first:
# reconst_raw = raw.copy()
# ica.apply(reconst_raw)
# raw.plot()
# reconst_raw.plot(block=True)

# # ------------------- Frequency analysis -------------------
# Plot frequency data with pyplot
# y, f = spectrum.get_data(return_freqs=True)
# y = 10*np.log10(y/1e-12) # Scale and convert to dB
# y = np.mean(y, axis=0) # Average over all EEG channels
# plt.plot(f, y)
# plt.show()
# raw = get_raw_from_xdf(paths[0])
# raw.crop(tmax=60).load_data()
# spectrum = raw.compute_psd()
# fig = spectrum.plot(average=True)
# preprocess(raw)
# spectrum = raw.compute_psd()
# fig = spectrum.plot(average=True)
# raw.plot(block=True)
    
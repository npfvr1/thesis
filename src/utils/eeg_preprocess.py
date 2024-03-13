import copy
import mne
from mne import Annotations
import pyxdf
import numpy as np


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
    # for i in range(eeg_channel_count + 1):
    #     if i == ref_channel:
    #         continue
    #     data[i] -= data[ref_channel]
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
    assert eeg_channel_count > 0
    if ref_electrode != "":
        assert ref_channel > -1
    for e in channel_names:
        assert e[0] in ["F", "C", "T", "P", "O"]
    assert sfreq > 0.0

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

    # TODO : Set the correct montage to use the electrodes' location

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
    Baseline correction is calculated from t = tmin to t = 0 and is applied.
    """
    events, events_id = mne.events_from_annotations(raw)
    print(events)
    return mne.Epochs(raw, events, event_id=events_id, preload=True, tmin=-12.5, tmax=25)

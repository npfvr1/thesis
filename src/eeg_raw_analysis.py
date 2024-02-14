import mne
import pyxdf
import numpy as np


"""
This script loads an XDF EEG data file, prints some information about it in the shell, and plots it using MNE.
A referential montage is used.
See diagram https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)#/media/File:EEG_10-10_system_with_additional_information.svg
"""


def get_raw_from_xdf(xdf_file_path: str, ref_electrode: str = "") -> mne.io.Raw:
    """[WIP] This function loads an XDF EEG file, and returns the corresponding Raw object to be used with the MNE package."""
    streams, _ = pyxdf.load_xdf(xdf_file_path)
    ref_channel = -1

    if streams[0]["time_series"].shape[0] > streams[1]["time_series"].shape[0]:
        stream_index = 0
    else:
        stream_index = 1

    # Count EEG channels and find the reference channel's index
    channels_info = streams[stream_index]["info"]["desc"][0]["channels"][0]["channel"]
    eeg_channel_count = 0
    for index, e in enumerate(channels_info):
        if e["type"][0] == "EEG":
            eeg_channel_count += 1
        if e["label"][0] == ref_electrode:
            ref_channel = index

    # Create Raw object for using MNE
    data = streams[stream_index]["time_series"].T
    for i in range(eeg_channel_count + 1):
        if i == ref_channel:
            continue
        data[i] -= data[ref_channel]
    # It is assumed that the EEG channels are the first ones
    data = data[:eeg_channel_count]
    # micro V to V and preamp gain ???
    data[:] *= 1e-6 #/ 2
    sfreq = float(streams[stream_index]["info"]["nominal_srate"][0])
    channel_names = [e["label"][0] + (' - ' + ref_electrode if e["label"][0] != ref_electrode else '') for e in channels_info[:eeg_channel_count]]

    # Data format check
    assert eeg_channel_count > 0
    assert ref_channel > -1
    for e in channel_names:
        assert e[0] in ["F", "C", "T", "P", "O"]
    assert sfreq > 0.0

    info = mne.create_info(channel_names, sfreq, ["eeg"] * eeg_channel_count)

    print("\nLoaded EEG file.\nInfo:")
    print("\tSampling rate: f =", sfreq, "Hz")
    print("\tEEG channel count: n =", eeg_channel_count, "\n")

    raw = mne.io.RawArray(data, info)
    if ref_electrode != "":
        raw.set_eeg_reference(ref_channels=[ref_electrode])

    return raw


mne.set_config('MNE_BROWSER_BACKEND', 'qt')
path = r"data\NIRS-EEG\Patient ID 7 - U5 (UWS)\Session 1\Post Administration 2\sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
ref_electrode = "Cz"
raw = get_raw_from_xdf(path, ref_electrode)
raw.crop(tmax=60).load_data()
raw.filter(l_freq=1, h_freq=70, verbose=False)
raw.notch_filter(np.arange(50, 250, 50), verbose=False)
spectrum = raw.compute_psd()
spectrum.plot(average=False, picks=['Fp1 - Cz'],exclude=[ref_electrode])
raw.plot(block=True)
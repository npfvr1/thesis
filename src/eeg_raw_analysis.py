import random
import mne
import pyxdf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from mne.preprocessing import ICA


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
    assert len(streams) == 2, "Unexpected XDF data structure"
    if streams[1]["time_series"].shape[0] > streams[0]["time_series"].shape[0]:
        stream_index = 1
        stream_index_markers = 0
    else :
        stream_index = 0
        stream_index_markers = 1

    # TODO : load markers and epoch the data
    # See https://mne.tools/stable/auto_tutorials/epochs/10_epochs_overview.html#sphx-glr-auto-tutorials-epochs-10-epochs-overview-py
    markers_time_stamps = streams[stream_index_markers]["time_series"]

    # Count EEG channels and find the reference channel's index
    channels_info = streams[stream_index]["info"]["desc"][0]["channels"][0]["channel"]
    eeg_channel_count = 0
    ref_channel = -1
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

    info = mne.create_info(channel_names, sfreq, ["eeg"] * eeg_channel_count)
    raw = mne.io.RawArray(data, info, verbose=False)

    if ref_electrode != "":
        raw.set_eeg_reference(ref_channels=[ref_electrode], verbose=False)
    else:
        raw.set_eeg_reference(verbose=False)  # Use the average montage

    return raw


def preprocess(raw: mne.io.Raw) -> None:
    """Preprocesses EEG data by filtering it : bandpass filter between 1 and 70 Hz, and notch filter at 50 Hz and some harmonics."""
    raw.filter(l_freq=1, h_freq=70, verbose=False)
    raw.notch_filter(np.arange(50, 250, 50), verbose=False)
    return


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


def test_xdf_files_reading() -> bool:
    paths = list()
    failed = list()

    # for path in tqdm(Path('data').rglob('*.xdf')):
    #     paths.append(path)

    paths.append(r"data\NIRS-EEG\Patient ID 1 - U1 (UWS)\Session 1\Post-administration 2\sub-P001_ses-S001_task-Default_run-001_eeg.xdf")
    paths.append(r"data\NIRS-EEG\Patient ID 17 - M5 (MCS)\Session 1\Post Administration 2\sub-P001_ses-S001_task-Default_run-001_eeg.xdf")
    paths.append(r"data\NIRS-EEG\Patient ID 23 - U16 (UWS)\Session 2\Post Administration 2\sub-P001_ses-S001_task-Default_run-001_eeg_old317.xdf")
    paths.append(r"data\NIRS-EEG\Patient ID 27 - U19 (UWS)\Session 1\Baseline\sub-P001_ses-S001_task-Default_run-001_eeg.xdf")
    paths.append(r"data\NIRS-EEG\Patient ID 27 - U19 (UWS)\Session 1\Post Administration 1\sub-P001_ses-S001_task-Default_run-001_eeg_old336.xdf")
    paths.append(r"data\NIRS-EEG\Patient ID 27 - U19 (UWS)\Session 1\Post Administration 2\sub-P001_ses-S001_task-Default_run-001_eeg.xdf")
    paths.append(r"data\NIRS-EEG\Patient ID 31 - M10 (UWS)\Session 2\Post Administration 1\sub-P001_ses-S001_task-Default_run-001_eeg_old367.xdf")
    paths.append(r"data\NIRS-EEG\Patient ID 31 - M10 (UWS)\Session 2\Post Administration 2\sub-P001_ses-S001_task-Default_run-001_eeg.xdf")
    paths.append(r"data\NIRS-EEG\Patient ID 34 - U23 (UWS)\Session 2\Baseline\sub-P001_ses-S001_task-Default_run-001_eeg_old398.xdf")
    paths.append(r"data\NIRS-EEG\Patient ID 34 - U23 (UWS)\Session 2\Post Administration 1\sub-P001_ses-S001_task-Default_run-001_eeg_old399.xdf")
    paths.append(r"data\NIRS-EEG\Patient ID 41 - U26 (UWS)\Session 1\Baseline\sub-P001_ses-S001_task-Default_run-001_eeg_old455.xdf")
    paths.append(r"data\NIRS-EEG\Patient ID 8 - U6 (UWS)\Session 1\Post Administration 2\sub-P001_ses-S001_task-Default_run-001_eeg_old87.xdf")
    paths.append(r"data\NIRS-EEG\Patient ID 8 - U6 (UWS)\Session 1\Post Administration 2\sub-P001_ses-S001_task-Default_run-001_eeg_old88.xdf")

    for path in tqdm(paths):
        try:
            _ = get_raw_from_xdf(path)
        except:
            failed.append(path)

    print("Failed for file(s) located at :")
    for path in failed:
        print(path)
    
    return len(failed) == 0


def get_random_xdf_file() -> str:
    paths = list()

    for path in Path('data').rglob('*.xdf'):
        paths.append(path)

    return random.choice(paths)


print("\n"*2)

# assert test_xdf_files_reading()

mne.set_config("MNE_BROWSER_BACKEND", "qt")
paths = list()
paths.append(r"data\NIRS-EEG\Patient ID 50 - U32 (UWS)\Session 1\Baseline\sub-P001_ses-S001_task-Default_run-001_eeg_old517.xdf")
# data\NIRS-EEG\Patient ID 47 - M18 (MCS)\Session 1\Post Administration 1\sub-P001_ses-S001_task-Default_run-001_eeg_old497.xdf
# paths.append(get_random_xdf_file())

for path in paths:
    print("\nNow working with file", path, "\n")

    # Load and preprocess data
    raw = get_raw_from_xdf(path)
    raw.load_data()#.crop(tmin=129, tmax=149).load_data()

    preprocess(raw)

    ica = ICA(n_components=5, max_iter="auto", random_state=2000)
    ica.fit(raw)
    explained_var_ratio = ica.get_explained_variance_ratio(raw)
    for channel_type, ratio in explained_var_ratio.items():
        print(
            f"Fraction of {channel_type} variance explained by all components: " f"{ratio}"
        )
    
    ica.plot_sources(raw, show_scrollbars=False)
    # ica.plot_overlay(raw, exclude=[0], picks="eeg")

    ica.exclude = [0]  # indices chosen based on various plots above

    # ica.apply() changes the Raw object in-place, so let's make a copy first:
    reconst_raw = raw.copy()
    ica.apply(reconst_raw)

    raw.plot()
    reconst_raw.plot(block=True)

# plt.show()

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

print("\n"*2)

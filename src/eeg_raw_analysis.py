import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.eeg_preprocess import *
from utils.file_mgt import *


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
    """
    Tests the loading of certain XDF files as Raw objects.
    """
    paths = list()
    failed = list()

    # for path in tqdm(Path('data').rglob('*.xdf')):
    #     paths.append(path)

    paths.append(
        r"data\NIRS-EEG\Patient ID 1 - U1 (UWS)\Session 1\Post-administration 2\sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
    )
    paths.append(
        r"data\NIRS-EEG\Patient ID 17 - M5 (MCS)\Session 1\Post Administration 2\sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
    )
    paths.append(
        r"data\NIRS-EEG\Patient ID 23 - U16 (UWS)\Session 2\Post Administration 2\sub-P001_ses-S001_task-Default_run-001_eeg_old317.xdf"
    )
    paths.append(
        r"data\NIRS-EEG\Patient ID 27 - U19 (UWS)\Session 1\Baseline\sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
    )
    paths.append(
        r"data\NIRS-EEG\Patient ID 27 - U19 (UWS)\Session 1\Post Administration 1\sub-P001_ses-S001_task-Default_run-001_eeg_old336.xdf"
    )
    paths.append(
        r"data\NIRS-EEG\Patient ID 27 - U19 (UWS)\Session 1\Post Administration 2\sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
    )
    paths.append(
        r"data\NIRS-EEG\Patient ID 31 - M10 (UWS)\Session 2\Post Administration 1\sub-P001_ses-S001_task-Default_run-001_eeg_old367.xdf"
    )
    paths.append(
        r"data\NIRS-EEG\Patient ID 31 - M10 (UWS)\Session 2\Post Administration 2\sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
    )
    paths.append(
        r"data\NIRS-EEG\Patient ID 34 - U23 (UWS)\Session 2\Baseline\sub-P001_ses-S001_task-Default_run-001_eeg_old398.xdf"
    )
    paths.append(
        r"data\NIRS-EEG\Patient ID 34 - U23 (UWS)\Session 2\Post Administration 1\sub-P001_ses-S001_task-Default_run-001_eeg_old399.xdf"
    )
    paths.append(
        r"data\NIRS-EEG\Patient ID 41 - U26 (UWS)\Session 1\Baseline\sub-P001_ses-S001_task-Default_run-001_eeg_old455.xdf"
    )
    paths.append(
        r"data\NIRS-EEG\Patient ID 8 - U6 (UWS)\Session 1\Post Administration 2\sub-P001_ses-S001_task-Default_run-001_eeg_old87.xdf"
    )
    paths.append(
        r"data\NIRS-EEG\Patient ID 8 - U6 (UWS)\Session 1\Post Administration 2\sub-P001_ses-S001_task-Default_run-001_eeg_old88.xdf"
    )

    for path in tqdm(paths):
        try:
            _ = get_raw_from_xdf(path)
        except:
            failed.append(path)

    print("Failed for file(s) located at :")
    for path in failed:
        print(path)

    return len(failed) == 0


# assert test_xdf_files_reading()

mne.set_config("MNE_BROWSER_BACKEND", "qt")
paths = list()
paths = get_random_xdf_file_paths(10)
# paths.append(
#     r"data\raw\NIRS-EEG\Patient ID 2 - U2 (UWS)\Session 2\Post Administration 2\sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
# )
# paths.append(
#     r"data\raw\NIRS-EEG\Patient ID 42 - U27 (UWS)\Session 2\Baseline\sub-P001_ses-S001_task-Default_run-001_eeg_old463.xdf"
# )

for path in paths:
    print("\nNow working with file", path, "\n")

    # ------ Part 1 : Raw + filter ----------------------------------------------------------------

    raw = get_raw_from_xdf(path)
    # raw.drop_channels("F10")
    # raw.set_eeg_reference()
    raw.load_data()
    raw = filter_raw(raw)
    raw.plot(n_channels=len(raw.ch_names), block=True)

    # ------ Part 2 : Epoched (split) + Artifact removal with ICA (manual selection) --------------

    epochs_baselined = epochs_from_raw(raw)
    epochs_baselined.load_data()
    #epochs_baselined.plot(events=True)

    del raw

    ica = ICA(max_iter="auto", random_state=2000)
    ica.fit(epochs_baselined)
    ica.plot_sources(epochs_baselined, show_scrollbars=False, block=True) # Set block to True to manually select the component to be removed
    #ica.exclude = [0]  # Or select the component's index here
    reconst_raw = epochs_baselined.copy()
    ica.apply(reconst_raw)
    # epochs_baselined.plot()
    # reconst_raw.plot()
    epochs_baselined = reconst_raw.copy()

    del reconst_raw

    from autoreject import get_rejection_threshold  # noqa

    # We can use the `decim` parameter to only take every nth time slice.
    # This speeds up the computation time. Note however that for low sampling
    # rates and high decimation parameters, you might not detect "peaky artifacts"
    # (with a fast timecourse) in your data. A low amount of decimation however is
    # almost always beneficial at no decrease of accuracy.
    reject = get_rejection_threshold(epochs_baselined)
    print('The rejection dictionary is %s' % reject)
    epochs_baselined.drop_bad(reject=reject)
    epochs_baselined.plot(block=True)

    continue

    # ------ Part 3 : Evoked (averaged) -----------------------------------------------------------

    evoked_audio = epochs_baselined["Audio"].average()
    evoked_audio.crop(tmin=0, tmax=10)
    # evoked_audio.plot()
    evoked_maths_1 = epochs_baselined["Mental arithmetics moderate"].average()
    evoked_maths_1.crop(tmin=0, tmax=25)
    # evoked_maths_1.plot()
    evoked_maths_2 = epochs_baselined["Mental arithmetics hard"].average()
    evoked_maths_2.crop(tmin=0, tmax=25)
    # evoked_maths_2.plot()

    del epochs_baselined

    spectrum_audio = evoked_audio.compute_psd(fmax=60)
    spectrum_audio.plot(average=True)

    spectrum_maths_1 = evoked_maths_1.compute_psd(fmax=60)
    spectrum_maths_1.plot(average=True)

    spectrum_maths_2 = evoked_maths_2.compute_psd(fmax=60)
    spectrum_maths_2.plot(average=True)

    plt.show()

    # from autoreject import AutoReject
    # ar = AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11,
    #                        n_jobs=1, verbose=True)
    # ar.fit(epochs[:20])  # fit on a few epochs to save time
    # epochs_ar, reject_log = ar.transform(epochs, return_log=True)


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

print("\n" * 2)

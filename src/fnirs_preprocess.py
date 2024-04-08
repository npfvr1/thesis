import mne

from utils.file_mgt import get_random_eeg_file_paths


paths = get_random_eeg_file_paths("snirf", 10)

for path in paths:

    raw_intensity = mne.io.read_raw_snirf(path)

    picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True)
    dists = mne.preprocessing.nirs.source_detector_distances(
        raw_intensity.info, picks=picks
    )
    raw_intensity.pick(picks[dists > 0.01])
    raw_intensity.plot()

    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
    raw_od.plot()

    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
    raw_haemo.plot()

    raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)
    raw_haemo.plot(n_channels=len(raw_haemo.ch_names), duration=700, show_scrollbars=False, block=True)
import logging
import os
import time

import mne
from tqdm import tqdm
import pandas as pd
from yasa import bandpower
import numpy as np
import antropy

from utils.file_mgt import *


"""Loads cleaned and epoched EEG data from .fif file.
Extracts features and saves them in a .csv file.
"""


def compute_brain_wave_band_power(epochs: mne.Epochs) -> tuple[float, float, float]:
    """
    Computes the relative band power averaged across channels and epochs, for Delta, Theta, and Alpha frequency bands.
    """
    delta_power = 0
    theta_power = 0
    alpha_power = 0

    epochs_data = epochs.get_data(copy=False)

    for epoch_id in range(epochs_data.shape[0]):

        df = bandpower(data = epochs_data[epoch_id] * 1e6,
                       sf = float(epochs._raw_sfreq[0]),
                       ch_names = epochs.ch_names,
                       relative = True,
                       bands = [(0.5, 4, "Delta"), (4, 8, "Theta"), (8, 13, "Alpha")] # Lower delta bound is 0.5 Hz so we use a 4s sliding window (default param)
                       )

        delta_power += np.mean(df[['Delta']].values)
        theta_power += np.mean(df[['Theta']].values)
        alpha_power += np.mean(df[['Alpha']].values)
        del df

    delta_power /= epochs_data.shape[0]
    theta_power /= epochs_data.shape[0]
    alpha_power /= epochs_data.shape[0]

    return (delta_power, theta_power, alpha_power)


def compute_entropies(epochs: mne.Epochs) -> tuple[float, float]:
    """
    Computes the permutation and spectral entropies averaged across channels and across epochs.
    """
    pe = 0
    se = 0
    epochs_data = epochs.get_data(copy=False)
    
    for epoch_id in range(epochs_data.shape[0]):
        for channel_id in range(epochs_data.shape[1]):
            x = epochs_data[epoch_id][channel_id]
            pe += antropy.perm_entropy(x, normalize=True)
            se += antropy.spectral_entropy(x, sf = epochs.info['sfreq'], method = 'welch', normalize = True)

    pe /= epochs_data.shape[1]
    se /= epochs_data.shape[1]

    pe /= epochs_data.shape[0]
    se /= epochs_data.shape[0]

    return (pe, se)


def main():

    # mne.set_config("MNE_BROWSER_BACKEND", "qt")
    mne.set_log_level("WARNING")
    logging.basicConfig(force=True, format='%(levelname)s - %(name)s - %(message)s', level=logging.INFO)

    paths = list()
    paths = get_random_eeg_file_paths("fif", 500)

    feature_list = []

    for path in tqdm(paths):

        logging.info("Current file is {}".format(path))
        features = []

        # ---- Recording info ----

        features.append(path.parts[3].replace("ID", "")) # Patient ID
        features.append(int(path.parts[2][-1])) # Drug
        features.append(0 if path.parts[4][-1] == "e" else int(path.parts[4][-1])) # Time of recording

        # ---- Epoch ----

        epochs = mne.read_epochs(path)
        
        # ---- Split by event type ----

        # epochs_audio = epochs['Audio']
        # audio_event_count = epochs_audio.selection.shape[0]
        epochs_arithmetics_moderate = epochs['Mental arithmetics moderate']
        arithmetics_moderate_event_count = epochs_arithmetics_moderate.selection.shape[0]
        epochs_arithmetics_hard = epochs['Mental arithmetics hard']
        arithmetics_hard_event_count = epochs_arithmetics_hard.selection.shape[0]
        del epochs
        # epochs_audio.crop(tmin=0, tmax=10)
        epochs_arithmetics_moderate.crop(tmin=0, tmax=25)
        epochs_arithmetics_hard.crop(tmin=0, tmax=25)

        # ---- Brain wave band power ----

        # powers_audio = compute_brain_wave_band_power(epochs_audio)
        powers_arithmetics_moderate = compute_brain_wave_band_power(epochs_arithmetics_moderate)
        powers_arithmetics_hard = compute_brain_wave_band_power(epochs_arithmetics_hard)

        powers = []

        # Weighted average (by event count)
        for i in range(3):
            temp_power = np.average([powers_arithmetics_moderate[i],
                                     powers_arithmetics_hard[i]],
                                     weights=[arithmetics_moderate_event_count, arithmetics_hard_event_count])
            powers.append(temp_power)

        # Alpha / delta
        powers.append(powers[2] / powers[0])

        features += powers

        # ---- Entropies ----

        # entropies_audio = compute_entropy_features(epochs_audio)
        entropies_arithmetics_moderate = compute_entropies(epochs_arithmetics_moderate)
        entropies_arithmetics_hard = compute_entropies(epochs_arithmetics_hard)

        entropies = []

        # Weighted average (by event count)
        for i in range(2):
            temp_entropy = np.average([entropies_arithmetics_moderate[i],
                                     entropies_arithmetics_hard[i]],
                                     weights=[arithmetics_moderate_event_count, arithmetics_hard_event_count])
            entropies.append(temp_entropy)

        features += entropies

        # ---- Save to data structure ----

        assert len(features) == 9
        feature_list.append(features)
    
    # ---- Save to file ----

    df = pd.DataFrame(feature_list, columns =['id', 'drug', 'time', 'delta', 'theta', 'alpha', 'ratio', 'pe', 'se'])
    df.to_csv(os.path.join("data", "processed", "eeg_features.csv"), index = False)


if __name__ == "__main__":
    t = time.time()
    main()
    logging.info("Script run in {} s".format(time.time() - t))

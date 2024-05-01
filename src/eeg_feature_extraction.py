import logging
import os
import time

import mne
from tqdm import tqdm
import pandas as pd
import antropy
from yasa import bandpower
import numpy as np

from utils.file_mgt import *


"""Loads cleaned and epoched EEG data from .fif file.
Extracts features and saves them in a .csv file.
"""


def compute_brain_wave_band_power(epochs: mne.Epochs) -> tuple[float, float, float, float, float]:
    """
    Computes the relative band power averaged across channels and epochs, for Delta, Theta, Alpha, Sigma, and Beta frequency bands.
    """
    delta_power = 0
    theta_power = 0
    alpha_power = 0
    sigma_power = 0
    beta_power = 0

    epochs_data = epochs.get_data()

    for epoch_id in range(epochs_data.shape[0]):

        df = bandpower(data = epochs_data[epoch_id] * 1e6,
                       sf = float(epochs._raw_sfreq[0]),
                       ch_names = epochs.ch_names,
                       relative = True)

        delta_power += np.mean(df[['Delta']].values)
        theta_power += np.mean(df[['Theta']].values)
        alpha_power += np.mean(df[['Alpha']].values)
        sigma_power += np.mean(df[['Sigma']].values)
        beta_power += np.mean(df[['Beta']].values)
        del df

    delta_power /= epochs_data.shape[0]
    theta_power /= epochs_data.shape[0]
    alpha_power /= epochs_data.shape[0]
    sigma_power /= epochs_data.shape[0]
    beta_power /= epochs_data.shape[0]

    return (delta_power, theta_power, alpha_power, sigma_power, beta_power)


def compute_entropy_features(epochs: mne.Epochs) -> tuple[float, float, float]:
    """
    Computes three complexity features (spectral and permutation entropies, and zero crossings) averaged across channels and across epochs.
    """
    se = 0
    pe = 0
    zc = 0
    
    epochs_data = epochs.get_data()
    
    for epoch_id in range(epochs_data.shape[0]):
        for channel_id in range(epochs_data.shape[1]):
            x = epochs_data[epoch_id][channel_id]
            se += antropy.spectral_entropy(x, sf = epochs.info['sfreq'], method = 'welch', normalize = True)
            pe += antropy.perm_entropy(x, normalize = True)
            zc += antropy.num_zerocross(x, normalize = True)

    se /= epochs_data.shape[1]
    pe /= epochs_data.shape[1]
    zc /= epochs_data.shape[1]

    se /= epochs_data.shape[0]
    pe /= epochs_data.shape[0]
    zc /= epochs_data.shape[0]

    return (se, pe, zc)


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

        epochs_audio = epochs['Audio']
        audio_event_count = epochs_audio.selection.shape[0]
        epochs_arithmetics_moderate = epochs['Mental arithmetics moderate']
        arithmetics_moderate_event_count = epochs_arithmetics_moderate.selection.shape[0]
        epochs_arithmetics_hard = epochs['Mental arithmetics hard']
        arithmetics_hard_event_count = epochs_arithmetics_hard.selection.shape[0]
        del epochs
        epochs_audio.crop(tmin=0, tmax=10)
        epochs_arithmetics_moderate.crop(tmin=0, tmax=25)
        epochs_arithmetics_hard.crop(tmin=0, tmax=25)

        # ---- Brain wave band power ----

        powers_audio = compute_brain_wave_band_power(epochs_audio)
        powers_arithmetics_moderate = compute_brain_wave_band_power(epochs_arithmetics_moderate)
        powers_arithmetics_hard = compute_brain_wave_band_power(epochs_arithmetics_hard)

        powers = []

        for i in range(5):
            # weighted mean of powers (weighted by even counts)
            temp_power = (audio_event_count * powers_audio[i]
                          + arithmetics_moderate_event_count * powers_arithmetics_moderate[i]
                          + arithmetics_hard_event_count * powers_arithmetics_hard[i]
                          ) / (audio_event_count
                               + arithmetics_moderate_event_count
                               + arithmetics_hard_event_count
                               )
            powers.append(temp_power)

        features += powers

        # ---- Entropy and nonlinear features ----

        entropies_audio = compute_entropy_features(epochs_audio)
        entropies_arithmetics_moderate = compute_entropy_features(epochs_arithmetics_moderate)
        entropies_arithmetics_hard = compute_entropy_features(epochs_arithmetics_hard)

        complexities = []

        for i in range(3):
            # weighted mean of complexitiesy measures (weighted by even counts)
            temp_complexity = (audio_event_count * entropies_audio[i]
                          + arithmetics_moderate_event_count * entropies_arithmetics_moderate[i]
                          + arithmetics_hard_event_count * entropies_arithmetics_hard[i]
                          ) / (audio_event_count
                               + arithmetics_moderate_event_count
                               + arithmetics_hard_event_count
                               )
            powers.append(temp_complexity)

        features += complexities

        # ---- AR process coefficients ----

        # for channel_index in range(evoked_audio.data.shape[0]):
        #     data = evoked_audio.data[channel_index]
        #     ar_coefficients, _ = sm.regression.yule_walker(data, order=10, method="mle") # TODO : hyperparameter (AR model order)
        #     features.extend(ar_coefficients[:5]) # TODO : hyperparameter (number of coefficients to keep)

        # ---- Save to data structure ----

        assert len(features) == 11
        feature_list.append(features)
    
    # ---- Save to file ----

    df = pd.DataFrame(feature_list, columns =['id', 'drug', 'time', 'delta', 'theta', 'alpha', 'sigma', 'beta', 'se', 'pe', 'zc'])
    df.to_csv(os.path.join("data", "processed", "eeg_features.csv"), index = False)


if __name__ == "__main__":
    t = time.time()
    main()
    logging.info("Script run in {} s".format(time.time() - t))

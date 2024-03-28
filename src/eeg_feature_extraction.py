import logging
import os
import time

import mne
from tqdm import tqdm
import pandas as pd
import antropy

from utils.file_mgt import *


"""Loads cleaned and epoched EEG data from .fif file.
Extracts features and saves them in a .csv file.
"""

def compute_brain_wave_band_power(epochs: mne.Epochs) -> tuple[float, float, float]:
    """
    Goes through all epochs and integrate the PSD using a step function for each channel.
    Returns
    -------
        The total delta, theta, and alpha band powers.
    """
    delta_power = 0
    theta_power = 0
    alpha_power = 0
    
    spectrum = epochs.compute_psd(fmax=30)
    for event_id in range(spectrum._data.shape[0]):
        for channel_id in range(spectrum._data[event_id].shape[0]):
            temp_freq = 0
            for freq, power in zip(spectrum.freqs, spectrum._data[event_id][channel_id]):
                delta_freq = freq - temp_freq
                if 1 <= freq and freq < 4:
                    delta_power += power * delta_freq
                elif 4 <= freq and freq < 8:
                    theta_power += power * delta_freq
                elif 8 <= freq and freq < 12:
                    alpha_power += power * delta_freq
                elif freq >= 12:
                    break
                temp_freq = freq

    return (delta_power, theta_power, alpha_power)


def compute_entropy_features(epochs: mne.Epochs) -> tuple[float, float, float]:
    """
    Goes through all epochs and compute the normalized spectral entropy, permutation entropy, and number of zero crsossings.
    (Per channel)
    """
    se = 0
    pe = 0
    zc = 0
    
    for event_id in range(epochs._data.shape[0]):
        for channel_id in range(epochs._data[event_id].shape[0]):
            x = epochs._data[event_id][channel_id]
            se += antropy.spectral_entropy(x, sf = epochs.info['sfreq'], method = 'welch', normalize = True)
            pe += antropy.perm_entropy(x, normalize = True)
            zc += antropy.num_zerocross(x, normalize = True)

    se /= epochs._data[event_id].shape[0]
    pe /= epochs._data[event_id].shape[0]
    zc /= epochs._data[event_id].shape[0]

    return (se, pe, zc)


def main():

    # mne.set_config("MNE_BROWSER_BACKEND", "qt")
    mne.set_log_level("WARNING")
    logging.basicConfig(force=True, format='%(levelname)s - %(name)s - %(message)s', level=logging.INFO)

    paths = list()
    paths = get_random_eeg_file_paths("fif", 500)
    # paths = get_random_eeg_file_paths_grouped_by_session("fif", 5)
    # paths.append(r"")

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
        # epochs.plot(block=True, events=True)

        # # Visualize the data distributions
        # for channel_index in range(10):#evoked_audio.data.shape[0]):
        #     data = evoked_audio.data[channel_index]
        #     plt.figure()
        #     plt.hist(data*1e6, color='gold', ec='black', bins=50)
        #     plt.title("Channel {}".format(channel_index))
        # plt.show()
        
        # ---- Split by event type ----

        epochs_audio = epochs['Audio']
        epochs_arithmetics_moderate = epochs['Mental arithmetics moderate']
        epochs_arithmetics_hard = epochs['Mental arithmetics hard']
        event_count = len(epochs.selection)
        del epochs
        epochs_audio.crop(tmin=0, tmax=10)
        epochs_arithmetics_moderate.crop(tmin=0, tmax=25)
        epochs_arithmetics_hard.crop(tmin=0, tmax=25)

        # ---- Brain wave band power ----

        # TODO : hyperparameter (choice of using all event types or only some)
        powers_audio = compute_brain_wave_band_power(epochs_audio)
        powers_arithmetics_moderate = compute_brain_wave_band_power(epochs_arithmetics_moderate)
        powers_arithmetics_hard = compute_brain_wave_band_power(epochs_arithmetics_moderate)
        
        delta_power = powers_audio[0] + powers_arithmetics_moderate[0] + powers_arithmetics_hard[0]
        delta_power /= event_count
        features.append(delta_power)
        theta_power = powers_audio[1] + powers_arithmetics_moderate[1] + powers_arithmetics_hard[1]
        theta_power /= event_count
        features.append(theta_power)
        alpha_power = powers_audio[2] + powers_arithmetics_moderate[2] + powers_arithmetics_hard[2]
        alpha_power /= event_count
        features.append(alpha_power)

        # ---- Entropy and nonlinear features ----

        entropies_audio = compute_entropy_features(epochs_audio)
        entropies_arithmetics_moderate = compute_entropy_features(epochs_arithmetics_moderate)
        entropies_arithmetics_hard = compute_entropy_features(epochs_arithmetics_moderate)

        se = entropies_audio[0] + entropies_arithmetics_moderate[0] + entropies_arithmetics_hard[0]
        se /= event_count
        features.append(se)
        pe = entropies_audio[1] + entropies_arithmetics_moderate[1] + entropies_arithmetics_hard[1]
        pe /= event_count
        features.append(pe)
        zc = entropies_audio[2] + entropies_arithmetics_moderate[2] + entropies_arithmetics_hard[2]
        zc /= event_count
        features.append(zc)

        # ---- AR process coefficients ----

        # for channel_index in range(evoked_audio.data.shape[0]):
        #     data = evoked_audio.data[channel_index]
        #     ar_coefficients, _ = sm.regression.yule_walker(data, order=10, method="mle") # TODO : hyperparameter (AR model order)
        #     features.extend(ar_coefficients[:5]) # TODO : hyperparameter (number of coefficients to keep)

        # ---- Save to data structure ----

        assert len(features) == 9
        feature_list.append(features)
    
    # ---- Save to file ----

    df = pd.DataFrame(feature_list, columns =['id', 'drug', 'time', 'delta', 'theta', 'alpha', 'se', 'pe', 'zc'])
    df.to_csv(os.path.join("data", "processed", "eeg_features4.csv"), index = False)


if __name__ == "__main__":
    t = time.time()
    main()
    logging.info("Script run in {} s".format(time.time() - t))

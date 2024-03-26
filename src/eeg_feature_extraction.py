import logging
import os
import time

import mne
from tqdm import tqdm
import numpy as np
import pandas as pd

from utils.file_mgt import *


"""Loads cleaned and epoched EEG data from .fif file.
Extracts features and saves them in a .csv file.
"""


def main():

    logging.basicConfig(force=True, format='%(levelname)s - %(name)s - %(message)s')
    # mne.set_config("MNE_BROWSER_BACKEND", "qt")
    mne.set_log_level("WARNING")

    paths = list()
    paths = get_random_eeg_file_paths("fif", 5)
    # paths = get_random_eeg_file_paths_one_session("fif")
    # paths.append(r"")

    feature_set = {} # Key: recording ID, Value: feature vector

    for path in tqdm(paths):

        logging.info("Current file is {}".format(path))
        features = {}

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
        # integrate the PSD using step function, for each (event, channel)

        delta_power = 0
        theta_power = 0
        alpha_power = 0
        for e in [epochs_audio, epochs_arithmetics_moderate, epochs_arithmetics_hard]:
            spectrum = e.compute_psd(fmax=30)
            # _ = spectrum.plot()
            # add_brain_wave_types_lines_on_pyplot_figure()
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
                # event_id += (8 if (i == 2) else (i * 3))
                # features['delta power for event {}'.format(event_id)] = theta_power # / number of channels ?
                # features['alpha power for event {}'.format(event_id)] = alpha_power # / number of channels ?

        features['delta'] = delta_power / event_count
        features['theta'] = theta_power / event_count
        features['alpha'] = alpha_power / event_count
        
        # plt.show()

        # # AR process coefficients
        # for channel_index in range(evoked_audio.data.shape[0]):
        #     data = evoked_audio.data[channel_index]
        #     ar_coefficients, _ = sm.regression.yule_walker(data, order=10, method="mle") # TODO : hyperparameter (AR model order)
        #     features.extend(ar_coefficients[:5]) # TODO : hyperparameter (number of coefficients to keep)

        data_point = path.parts[2] + "-" + path.parts[3] + "-" + path.parts[4]
        feature_set[data_point] = features
    
    df = pd.DataFrame.from_dict(feature_set, orient="index")
    df.to_csv(os.path.join("data", "processed", "eeg_features.csv"))


if __name__ == "__main__":
    t = time.time()
    main()
    logging.info("Script run in {} s".format(time.time() - t))

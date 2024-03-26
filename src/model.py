import os

import pandas as pd
import numpy as np


# ------ Part 1 : Load and concatenate data -------------------------------------------------------

df = pd.read_csv(os.path.join("data", "processed", "eeg_features.csv"))
eeg_data = df.to_numpy()
del df

# df = pd.read_csv(os.path.join("data", "processed", "fnirs_features.csv"))
# fnirs_data = df.to_numpy()
# del df

# df = pd.read_csv(os.path.join("data", "processed", "pupillometry_features.csv")) # Make sure the keys are the same for pupillometry data
# pupillometry_data = df.to_numpy()
# del df

# data = np.concatenate((eeg_data, fnirs_data, pupillometry_data), axis=?)

# ------ Part 2 : Normalize the data --------------------------------------------------------------
# Reduce dimensionality (e.g. with PCA) before/after normalizing?
# See Google tutorial on selecting the normalization technique

# ------ Part 3 : Clustering ----------------------------------------------------------------------
# TODO : set up mlops pipeline for reproducibility of experiments

# ------ Part 4 : Results -------------------------------------------------------------------------
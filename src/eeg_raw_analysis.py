import os

import numpy as np
import mne
import pyxdf


"""
This script loads an XDF EEG data file, prints some information about it in the shell, and plots it using MNE.
A referential montage is used with reference electrode REF_CHANNEL (see below).
See diagram https://en.wikipedia.org/wiki/10%E2%80%9320_system_(EEG)#/media/File:EEG_10-10_system_with_additional_information.svg
"""


PATH = os.path.join(
    os.getcwd(),
    "data",
    "NIRS-EEG",
    "Patient ID 1 - U1 (UWS)",
    "Session 2",
    "Baseline",
    "sub-P001_ses-S001_task-Default_run-001_eeg_old15.xdf",
)
REF_ELECTRODE = "Cz"
REF_CHANNEL = -1

streams, header = pyxdf.load_xdf(PATH)

# Count EEG channels and find the reference channel's index
channels_info = streams[0]["info"]["desc"][0]["channels"][0]["channel"]
eeg_channel_count = 0
for index, e in enumerate(channels_info):
    if e["type"][0] == "EEG":
        eeg_channel_count += 1
    if e["label"][0] == REF_ELECTRODE:
        REF_CHANNEL = index
if eeg_channel_count == 0 or REF_CHANNEL == -1:
    raise RuntimeError("There is a problem with the file.")

# Create Raw object for using MNE
data = streams[0]["time_series"].T
for i in range(eeg_channel_count + 1):
    if i == REF_CHANNEL:
        continue
    data[i] -= data[REF_CHANNEL]
# It is assumed that the EEG channels are the first ones
data = data[: eeg_channel_count + 1]
# micro V to V and preamp gain ???
data[: eeg_channel_count + 1] *= 1e-6 / 2
# Drop ref channel
data = np.concatenate((data[:REF_CHANNEL], data[REF_CHANNEL + 1 :]))

sfreq = float(streams[0]["info"]["nominal_srate"][0])
info = mne.create_info(eeg_channel_count, sfreq, ["eeg"] * eeg_channel_count)
raw = mne.io.RawArray(data, info)
raw.plot(block=True)

print("\nLoaded EEG file.\nInfo:")
print("\tSampling rate: f =", sfreq, "Hz")
print("\tEEG channel count: n =", eeg_channel_count)
print("\tFirst channel:", channels_info[00])

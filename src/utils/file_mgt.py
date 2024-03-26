import random
import os
from pathlib import Path


def get_random_eeg_file_paths(extension: str, count: int = 1) -> list[str]:
    """
    Returns a list containing the relative paths of different random EEG files located within the 'data/raw' folder.

    Parameters
    ----------
    extension: str
        The file extension to look for ("xdf" or "fif").
    count: int
        The number of paths to return.
    """
    if count < 1:
        return [""]
    
    if extension not in ["xdf", "fif"]:
        raise ValueError("Parameter extension must be 'xdf' or 'fif'")

    paths = list()

    for path in Path(os.path.join("data", "raw")).rglob("*." + extension):
        paths.append(path)

    if count > len(paths):
        return paths

    return random.sample(paths, count)


def get_random_eeg_file_paths_one_session(extension: str) -> list[str]:
    """
    Get the three EEG files making up one session of a random patient.
    
    Parameters
    ----------
    extension: str
        The file extension to look for ("xdf" or "fif").
    """
    
    if extension not in ["xdf", "fif"]:
        raise ValueError("Parameter extension must be 'xdf' or 'fif'")
    
    paths = list()

    while len(paths) != 3:
        paths.clear()
        temp = os.path.join("data", "raw")
        folder = random.choice(os.listdir(temp)) # Random drug
        temp = os.path.join(temp, folder)
        folder = random.choice(os.listdir(temp)) # Random patient and session
        temp = os.path.join(temp, folder)
        for path in Path(temp).rglob("*." + extension):
            paths.append(path)

    return paths

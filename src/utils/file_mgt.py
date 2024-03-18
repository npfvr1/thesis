import random
import os
from pathlib import Path


def get_random_xdf_file_paths(count: int, seed:float) -> list[str]:
    """
    Returns a list containing the relative paths of different random .xdf files located within the 'data/raw' folder.

    Parameters
    ----------
    count: int
        The number of paths to return. If it is less than one, or greater than the total number of files, all the paths are returned.
    seed: float
        The random seed to use.
    """
    if count < 1:
        return [""]

    paths = list()
    random.seed(seed)

    for path in Path(os.path.join("data", "raw")).rglob("*.xdf"):
        paths.append(path)

    if count > len(paths):
        return paths

    return random.sample(paths, count)


def get_random_xdf_file_paths_one_session(seed:float) -> list[str]:
    """
    Get the three EEG recordings making up one session of a random patient.
    
    Parameters
    ----------
    seed: float
        The random seed to use.
    """
    paths = list()
    random.seed(seed)

    while len(paths) != 3:
        paths.clear()
        temp = os.path.join("data", "raw")
        folder = random.choice(os.listdir(temp)) # Random drug
        temp = os.path.join(temp, folder)
        folder = random.choice(os.listdir(temp)) # Random patient
        temp = os.path.join(temp, folder)
        folder = random.choice(os.listdir(temp)) # Random session
        temp = os.path.join(temp, folder)
        for path in Path(temp).rglob("*.xdf"):
            paths.append(path)

    return paths

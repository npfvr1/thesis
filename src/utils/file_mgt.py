import random
import os
from pathlib import Path


def get_random_eeg_file_paths(extension: str, count: int = 1) -> list[str]:
    """
    Get random EEG files.

    Parameters
    ----------
    extension: str
        The file extension to look for ("xdf" or "fif").
    count: int
        The number of paths to return.
    """
    
    if count < 1 or extension not in ["xdf", "fif", "snirf"]:
        raise ValueError("Invalid parameters")

    paths = list()

    for path in Path(os.path.join("data", "raw")).rglob("*." + extension):
        paths.append(path)

    if count >= len(paths):
        return paths

    return random.sample(paths, count)


def get_random_eeg_file_paths_grouped_by_session(extension: str, session_count: int = 1) -> list[list[str]]:
    """
    Get random sets of EEG files making up one session.
    
    Parameters
    ----------
    extension: str
        The file extension to look for ("xdf" or "fif").
    session_count: int
        The number of sessions to return.
    """
    
    if session_count < 1 or extension not in ["xdf", "fif"]:
        raise ValueError("Invalid parameters")
    
    paths = list()

    main_path = os.path.join("data", "raw")

    for drug_folder in [f.path for f in os.scandir(main_path) if f.is_dir()]:

        for session_folder in [f.path for f in os.scandir(drug_folder) if f.is_dir()]:

            temp_paths = list()

            for path in Path(session_folder).rglob("*." + extension):
                temp_paths.append(path)
            paths.append(temp_paths)

    if session_count >= len(paths):
        return paths

    return random.sample(paths, session_count)

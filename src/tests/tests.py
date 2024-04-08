import os
import sys

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory by going one level up
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path
sys.path.append(parent_dir)

from tqdm import tqdm

from eeg_preprocess import get_raw_from_xdf
from utils.file_mgt import get_random_eeg_file_paths


def test_xdf_files_reading() -> bool:
    """
    Tests the loading of XDF files as Raw objects.
    """
    paths = get_random_eeg_file_paths("xdf", 500)
    failed = list()

    for path in tqdm(paths):
        try:
            _ = get_raw_from_xdf(path)
        except:
            failed.append(path)

    print("Failed for {} file(s) located at :".format(len(failed)))
    for path in failed:
        print(path)

    return len(failed) == 0

assert test_xdf_files_reading()
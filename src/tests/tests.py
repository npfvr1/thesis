from pathlib import Path

from tqdm import tqdm

from eeg_preprocess import get_raw_from_xdf


def test_xdf_files_reading() -> bool:
    """
    Tests the loading of XDF files as Raw objects.
    """
    paths = list()
    failed = list()

    for path in tqdm(Path('data').rglob('*.xdf')):
        paths.append(path)

    for path in tqdm(paths):
        try:
            _ = get_raw_from_xdf(path)
        except:
            failed.append(path)

    print("Failed for file(s) located at :")
    for path in failed:
        print(path)

    return len(failed) == 0

assert test_xdf_files_reading()
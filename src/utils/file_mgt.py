
import random
from pathlib import Path


def get_random_xdf_file_paths(count: int) -> list[str]:
    """
    Returns a list containing the relative paths of different random .xdf files located within the 'data/raw' folder.
    """
    if count < 1:
        return [""]

    paths = list()

    for path in Path("data/raw").rglob("*.xdf"):
        paths.append(path)

    if count > len(paths):
        return paths

    return random.sample(paths, count)

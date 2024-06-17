import shutil
from pathlib import Path

import requests


def download_files(url: str, output_directory: Path):
    """Downloads a file from a given URL to a given
    output directory.

    Parameters
    ----------
    url : str
        URL of the file.
    output_directory : Path
        Output directory.
    """
    if not isinstance(url, str):
        raise TypeError("Expected type str for url!")

    filename = Path(url.split("/")[-1])
    filename = output_directory / filename

    if filename.is_file():
        return

    with requests.get(url, stream=True) as r:
        with open(filename, "wb") as f:
            shutil.copyfileobj(r.raw, f)

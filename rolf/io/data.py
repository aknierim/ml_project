import shutil
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
import requests
import tomli
from astropy.table import QTable
from joblib import Parallel, delayed
from rich.progress import Progress

ROOT = Path(__file__).parents[2].resolve()


class GetData:
    def __init__(self, output_directory) -> None:
        self.output_directory = output_directory

    def from_name(self) -> None:
        with open(ROOT / "urls.toml", "rb") as f:
            urls = tomli.load(f)

        with Progress() as progress:
            dl = progress.add_task("[red]Downloading...", total=len(urls))

            Parallel(backend="threading", n_jobs=16)(
                delayed(self._from_name)(file, url, self.output_directory, progress, dl)
                for (file, url) in urls.items()
            )

    def _from_name(
        self,
        filename: str | Path,
        url: str,
        output_directory: Path,
        progress: Progress,
        task: int,
    ) -> None:
        """ """
        if not isinstance(url, str):
            raise TypeError("Expected type str for url!")

        if not isinstance(filename, Path):
            filename = Path(filename)

        self._download(url, filename)

        progress.update(task, advance=1)

    def from_url(self, url: str):
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
        self._download(url, filename)

    def _download(self, url, filename) -> None:
        filename = self.output_directory / filename

        if filename.is_file():
            return
        with requests.get(url, stream=True) as r:
            with open(filename, "wb") as f:
                shutil.copyfileobj(r.raw, f)


def read_hdf5(filepath: str | Path):
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    with h5py.File(filepath, "r") as file:
        idx = []
        entries = []
        ra = []
        dec = []
        sources = []
        filepaths = []
        labels = []
        splits = []

        for i, key in enumerate(file.keys()):
            data_entry = file[key + "/Img"]
            label_entry = np.array(file[key + "/Label_literature"], dtype=int)
            split_entry = np.array(file[key + "/Split_literature"], dtype=str)

            ra_attr = np.array(data_entry.attrs["RA"]) * u.deg
            dec_attr = np.array(data_entry.attrs["DEC"]) * u.deg
            source_attr = np.array(data_entry.attrs["Source"], dtype=str)
            filepath_attr = np.array(data_entry.attrs["Filepath_literature"], dtype=str)

            data_entry = np.array(data_entry)

            idx.append(i)
            entries.append(data_entry)
            ra.append(ra_attr)
            dec.append(dec_attr)
            sources.append(source_attr)
            labels.append(filepath_attr)
            filepaths.append(label_entry)
            splits.append(split_entry)

        table = QTable(
            [idx, entries, ra, dec, sources, filepaths, labels, splits],
            names=("index", "img", "RA", "DEC", "source", "filepath", "label", "split"),
        )

        del idx, entries, ra, dec, sources, filepaths, labels, splits

    return table

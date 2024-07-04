import shutil
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
import pandas as pd
import requests
import tomli
from astropy.table import QTable
from joblib import Parallel, delayed
from rich.progress import Progress
from sklearn.model_selection import train_test_split
from torch import FloatTensor
from torch.utils.data import Dataset
from torchvision.io import read_image

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


class CreateTorchDataset(Dataset):
    def __init__(
        self,
        img_labels: list[int],
        img_filepaths: list[str],
        img_dir: str | Path,
        transform=None,
        target_transform=None,
    ):
        if not isinstance(img_dir, Path):
            img_dir = Path(img_dir)

        self.img_labels = img_labels
        self.img_filepaths = img_filepaths
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_dir / self.img_filepaths[idx]

        image = read_image(img_path)
        label = self.img_labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image.type(FloatTensor), label


def read_hdf5(filepath: str | Path):
    if not isinstance(filepath, Path):
        filepath = Path(filepath)

    idx = []
    entries = []
    ra = []
    dec = []
    sources = []
    filepaths = []
    labels = []
    splits = []

    with h5py.File(filepath, "r") as file:
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
            filepaths.append(filepath_attr)
            labels.append(label_entry)
            splits.append(split_entry)

    table = QTable(
        [idx, entries, ra, dec, sources, filepaths, labels, splits],
        names=("index", "img", "RA", "DEC", "source", "filepath", "label", "split"),
    )

    del idx, entries, ra, dec, sources, filepaths, labels, splits

    return table


class ReadHDF5:
    def __init__(
        self,
        filepath: str | Path,
        validation_ratio: float = None,
        test_ratio: float = None,
        random_state: int = 42,
    ) -> pd.DataFrame | QTable:
        """Reads HDF5 file and creates dataset."""
        if not isinstance(filepath, Path):
            filepath = Path(filepath)

        self.filepath = filepath

        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

    def get_full_data(self):
        idx = []
        entries = []
        ra = []
        dec = []
        sources = []
        filepaths = []
        labels = []
        splits = []

        with h5py.File(self.filepath, "r") as file:
            for i, key in enumerate(file.keys()):
                data_entry = file[key + "/Img"]
                label_entry = np.array(file[key + "/Label_literature"], dtype=int)
                split_entry = np.array(file[key + "/Split_literature"], dtype=str)

                ra_attr = np.array(data_entry.attrs["RA"]) * u.deg
                dec_attr = np.array(data_entry.attrs["DEC"]) * u.deg
                source_attr = np.array(data_entry.attrs["Source"], dtype=str)
                filepath_attr = np.array(
                    data_entry.attrs["Filepath_literature"], dtype=str
                )

                data_entry = np.array(data_entry)

                idx.append(i)
                entries.append(data_entry)
                ra.append(ra_attr)
                dec.append(dec_attr)
                sources.append(source_attr)
                filepaths.append(filepath_attr)
                labels.append(label_entry)
                splits.append(split_entry)

        table = QTable(
            [idx, entries, ra, dec, sources, filepaths, labels, splits],
            names=(
                "index",
                "img",
                "RA",
                "DEC",
                "source",
                "filepaths",
                "label",
                "split",
            ),
        )

        if self.validation_ratio and self.test_ratio:
            table["labels"] = self._get_splits(table)

        del idx, entries, ra, dec, sources, filepaths, labels, splits

        return table

    def get_labels_and_paths(self) -> pd.DataFrame:
        filepaths = []
        labels = []
        splits = []

        with h5py.File(self.filepath, "r") as file:
            for i, key in enumerate(file.keys()):
                label_entry = np.array(file[key + "/Label_literature"], dtype=int)
                split_entry = np.array(file[key + "/Split_literature"], dtype=str)

                filepath_attr = np.array(
                    file[key + "/Img"].attrs["Filepath_literature"], dtype=str
                )

                filepaths.append(filepath_attr)
                labels.append(label_entry)
                splits.append(split_entry)

        df = pd.DataFrame({"filepaths": filepaths, "labels": labels, "splits": splits})

        if self.validation_ratio and self.test_ratio:
            df["labels"] = self._get_splits(df)

        return df

    def _get_splits(self, data):
        X = data["filepaths"]
        y = data["labels"]

        indices = np.arange(len(y))

        X_temp, _, y_temp, y_test, temp_idx, test_idx = train_test_split(
            X,
            y,
            indices,
            test_size=self.test_ratio,
            shuffle=True,
            random_state=self.random_state,
        )
        _, _, y_train, y_val, train_idx, val_idx = train_test_split(
            X_temp,
            y_temp,
            temp_idx,
            test_size=self.validation_ratio,
            shuffle=True,
            random_state=self.random_state,
        )

        y = np.concatenate((y_train, y_val, y_test))
        idx = np.concatenate((train_idx, val_idx, test_idx))

        df = pd.DataFrame({"y": y, "idx": idx})
        df = df.sort_values(by="idx")

        labels = df["y"].to_numpy()

        return labels

    def create_torch_dataset(
        self,
        img_dir: str | Path,
        transform=None,
        target_transform=None,
    ) -> tuple:
        data = self.get_labels_and_paths()

        train = data[data["split"] == "train"]
        test = data[data["split"] == "test"]
        valid = data[data["split"] == "valid"]

        train_set = CreateTorchDataset(
            train["label"].to_numpy(),
            train["filepath"].to_numpy(),
            img_dir=img_dir,
        )
        test_set = CreateTorchDataset(
            test["label"].to_numpy(), test["filepath"].to_numpy(), img_dir=img_dir
        )
        val_set = CreateTorchDataset(
            valid["label"].to_numpy(),
            valid["filepath"].to_numpy(),
            img_dir=img_dir,
        )

        return train_set, test_set, val_set

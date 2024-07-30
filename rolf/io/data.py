"""Data handling module."""

import shutil
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
import pandas as pd
import requests
import tomli
import torch
from astropy.table import QTable
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from rich.progress import Progress, track
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision.io import read_image
from torchvision.transforms import v2 as transforms

ROOT = Path(__file__).parents[2].resolve()


class GetData:
    """Data downloader class."""

    def __init__(self, output_directory: str | Path) -> None:
        """Initializes the class.

        Parameters
        ----------
        output_directory : Path or str
            Output directory for the data download.
        """
        self.output_directory = output_directory

    def from_name(self) -> None:
        """Downloads files from the URLs provided in the
        urls.toml file provided in the root directory of
        the source repository.
        """
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
        """Downloads a data file from a given URL.

        Parameters
        ----------
        filename : str or Path
            Name of the output file.
        url : str
            URL to download from.
        output_directory : str or Path
            Output directory.
        progress : rich.progress.Progress
            Progress bar instance.
        task : int
            Current progress bar task id.
        """
        if not isinstance(url, str):
            raise TypeError("Expected type str for url!")

        if not isinstance(filename, Path):
            filename = Path(filename)

        self._download(url, filename)

        progress.update(task, advance=1)

    def from_url(self, url: str) -> None:
        """Downloads a file from a given URL to a given
        output directory.

        Parameters
        ----------
        url : str
            URL of the file.
        """
        if not isinstance(url, str):
            raise TypeError("Expected type str for url!")

        filename = Path(url.split("/")[-1])
        self._download(url, filename)

    def _download(self, url: str, filename: str | Path) -> None:
        """Downloads files from given url
        and saves them to the output directory.

        Parameters
        ----------
        url : str
            URL to download from.
        filename : str or Path
            Output file name.
        """
        filename = self.output_directory / filename

        if filename.is_file():
            return
        with requests.get(url, stream=True) as r:
            with open(filename, "wb") as f:
                shutil.copyfileobj(r.raw, f)


class CreateTorchDataset(Dataset):
    """TorchDataset constructor class."""

    def __init__(
        self,
        img_labels: list[int],
        img_filepaths: list[str],
        img_dir: str | Path,
        transform=None,
        target_transform=None,
    ) -> None:
        """Initializes the class.

        Parameters
        ----------
        img_labels : list[int]
            Label data.
        img_filepaths : list[str]
            List of filepaths to the images.
        img_dir : str or Path
            Path to the image directory.
        transform : torchvision.transforms
            Transformation for the image dataset.
        target_transform : torchvision.transforms
            Transformation for the label data.
        """
        if not isinstance(img_dir, Path):
            img_dir = Path(img_dir)

        self.img_labels = img_labels
        self.img_filepaths = img_filepaths
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """Returns length of the dataset."""
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> tuple:
        """Item getter method.

        Parameters
        ----------
        idx : int
            Index of the image.

        Returns
        -------
        image : torch.Tensor
            Image array.
        label : torch.Tensor
            Label data.
        """
        img_path = self.img_dir / self.img_filepaths[idx]

        image = read_image(img_path)
        label = self.img_labels[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image.type(torch.FloatTensor), label


class ReadHDF5:
    """Reads HDF5 data file and creates torch Datasets
    and DataLoaders. Applys data splits and transformations.
    """

    def __init__(
        self,
        filepath: str | Path,
        validation_ratio: float = None,
        test_ratio: float = None,
        random_state: int = 42,
    ) -> None:
        """Reads HDF5 file and creates dataset.

        Parameters
        ----------
        filepath : str or Path
            Path to the HDF5 file.
        validation_ratio : float, optional
            Validation radio of the data split.
        test_ratio : float, optional
            Test ratio of the data split.
        random_state : int, optional
            Seed for the data split.
        """
        if not isinstance(filepath, Path):
            filepath = Path(filepath)

        self.filepath = filepath

        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

        self.data_keys = ["train", "valid", "test"]
        self.transformer = {}

        for key in self.data_keys:
            self.transformer[key] = None

    def get_full_data(self, progress: Progress = None, task: int = None) -> QTable:
        """Reads the full data from the hdf5 file.

        Parameters
        ----------
        progress : rich.progress.Progress
            Progress bar instance.
        task : TaskID
            Task ID of the current task.

        Returns
        -------
        table : astropy.table.QTable
            Table object.
        """
        idx = []
        entries = []
        ra = []
        dec = []
        sources = []
        filepaths = []
        labels = []
        splits = []

        if progress and track is not None:
            progress.console.print("[cyan][INFO][reset] Loading h5 file...")

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

        if progress and task is not None:
            progress.advance(task)
            progress.console.print("[cyan][INFO][reset] Creating data table object...")

        table = QTable(
            [idx, entries, ra, dec, sources, filepaths, labels, splits],
            names=(
                "index",
                "img",
                "RA",
                "DEC",
                "source",
                "filepath",
                "label",
                "split",
            ),
        )

        if progress and task is not None:
            progress.advance(task)

        if self.validation_ratio and self.test_ratio:
            if progress and task is not None:
                progress.console.print("[cyan][INFO][reset] Applying data split...")
            table["split"] = self._get_splits(table)

            if progress and task:
                progress.advance(task)

        del idx, entries, ra, dec, sources, filepaths, labels, splits

        self.df = table

        return table

    def get_labels_and_paths(self) -> pd.DataFrame:
        """Lightweight hdf5 reading method that only
        saves labels and image paths.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame containing labels, splits, and filepaths.
        """
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

        df = pd.DataFrame({"filepath": filepaths, "label": labels, "split": splits})

        if self.validation_ratio and self.test_ratio:
            df["split"] = self._get_splits(df)

        return df

    def _get_splits(self, data: pd.DataFrame) -> ArrayLike:
        """Gets splits via validation_ratio and test_ratio.

        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing filepaths, splits, and labels.

        Returns
        -------
        splits : array_like
            Array containing new splits.
        """
        X = data["filepath"]
        y = data["label"]

        valid_count = int(len(y) * self.validation_ratio)
        test_count = int(len(y) * self.test_ratio)
        train_count = len(y) - valid_count - test_count

        train, valid, test = random_split(X, [train_count, valid_count, test_count])

        idx = np.concatenate((train.indices, valid.indices, test.indices))

        train_splits = np.full_like(train.indices, "train", dtype="<U5")
        valid_splits = np.full_like(valid.indices, "valid", dtype="<U5")
        test_splits = np.full_like(test.indices, "test", dtype="<U5")

        split = np.concatenate((train_splits, valid_splits, test_splits))

        df = pd.DataFrame({"idx": idx, "split": split})
        df = df.sort_values(by="idx")

        splits = df["split"].to_numpy()

        return splits

    def make_transformer(self, progress: Progress = None, task: int = None) -> None:
        """Creates transformations for the dataset.

        Parameters
        ----------
        progress : rich.progress.Progress
            Progress bar instance.
        task : TaskID
            Task ID of the current task.
        """
        if progress and task is not None:
            _ = self.get_full_data(progress, task)
            progress.console.print(
                "[cyan][INFO][reset] Applying data transformations..."
            )
        else:
            _ = self.get_full_data()

        mean, std = {}, {}
        for label in self.data_keys:
            frame = self.df[self.df["split"] == label]["img"] / 255
            mean[label] = np.mean(frame)
            std[label] = np.std(frame)

        self.transformer["train"] = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=[mean["train"]], std=[std["train"]]),
            ]
        )

        self.transformer["valid"] = transforms.Compose(
            [
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=[mean["valid"]], std=[std["valid"]]),
            ]
        )

        self.transformer["test"] = transforms.Compose(
            [
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=[mean["test"]], std=[std["test"]]),
            ]
        )

        if progress and task is not None:
            progress.advance(task)

    def create_torch_datasets(
        self,
        img_dir: str | Path,
        transform: dict = {},
        target_transform=None,
    ) -> tuple:
        """Creates torch Datasets.

        Parameters
        ----------
        img_dir : str or Path
            Base directory of the images.
        transform : dict
            Dictionary of transformations.
        target_transform : torchvision.transforms
            Transformation for the label data.

        Returns
        -------
        tuple
            Tuple of torch Datasets.
        """
        data = self.get_labels_and_paths()

        train = data[data["split"] == "train"]
        valid = data[data["split"] == "valid"]
        test = data[data["split"] == "test"]

        class_sample_count = np.unique(train["label"], return_counts=True)[1]
        self.train_weight = 1.0 / class_sample_count
        samples_weight = np.array([self.train_weight[t] for t in train["label"]])
        samples_weight = torch.from_numpy(samples_weight)

        self.sampler = WeightedRandomSampler(
            samples_weight.type("torch.DoubleTensor"), len(samples_weight)
        )

        if len(transform.keys()) != 0:
            for key in self.data_keys:
                try:
                    self.transformer[key] = transform[key]
                except KeyError:
                    continue

        train_imgs = train["filepath"].to_numpy()
        valid_imgs = valid["filepath"].to_numpy()
        test_imgs = test["filepath"].to_numpy()

        self.train_set = CreateTorchDataset(
            train["label"].to_numpy(),
            train_imgs,
            img_dir=img_dir,
            transform=self.transformer["train"],
        )
        self.valid_set = CreateTorchDataset(
            valid["label"].to_numpy(),
            valid_imgs,
            img_dir=img_dir,
            transform=self.transformer["valid"],
        )
        self.test_set = CreateTorchDataset(
            test["label"].to_numpy(),
            test_imgs,
            img_dir=img_dir,
            transform=self.transformer["test"],
        )

        return self.train_set, self.valid_set, self.test_set

    def create_data_loaders(
        self,
        batch_size: int,
        img_dir: str | Path,
        train_set: Dataset = None,
        valid_set: Dataset = None,
        test_set: Dataset = None,
        sampler: torch.utils.data.sampler.Sampler = None,
    ) -> tuple:
        """Creates torch DataLoaders.

        Parameters
        ----------
        batch_size : int
            Batch size.
        img_dir : str or Path
            Base directory of the images.
        train_set : torch.Dataset
            Training dataset.
        valit_set : torch.Dataset
            Validation dataset.
        test_set : torch.Dataset
            Test dataset.
        sampler: torch.utils.data.sampler.Sampler, optional
            Data sampler.

        Returns
        -------
        tuple
            Tuple of torch DataLoaders.
        """
        try:
            train_set = self.train_set
            valid_set = self.valid_set
            test_set = self.test_set
        except AttributeError:
            pass

        if None in (train_set, valid_set, test_set):
            train_set, valid_set, test_set = self.create_torch_datasets(img_dir)

        try:
            sampler = self.sampler
        except AttributeError:
            pass

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True if sampler is None else False,
            drop_last=True,
            pin_memory=True,
            num_workers=4,
            sampler=sampler,
        )
        valid_loader = DataLoader(
            valid_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
        )

        return train_loader, valid_loader, test_loader

    def create_rf_data(
        self,
        img_dir: str | Path,
        train_set: Dataset = None,
        valid_set: Dataset = None,
        test_set: Dataset = None,
        sampler: torch.utils.data.sampler.Sampler = None,
    ) -> tuple:
        """Creates datasets for the random forest classifier.
        img_dir : str or Path
            Base directory of the images.
        train_set : torch.Dataset
            Training dataset.
        valit_set : torch.Dataset
            Validation dataset.
        test_set : torch.Dataset
            Test dataset.
        sampler: torch.utils.data.sampler.Sampler, optional
            Data sampler.

        Returns
        -------
        tuple
            Tuple of X_train, X_valid, X_test,
            y_train, y_valid, and y_test datasets.

        """
        train, valid, test = self.create_data_loaders(
            batch_size=10,
            img_dir=img_dir,
            train_set=train_set,
            valid_set=valid_set,
            test_set=test_set,
            sampler=sampler,
        )

        train = list(iter(train))
        valid = list(iter(valid))
        test = list(iter(test))

        X_train = np.concatenate([train[i][0] for i in range(len(train))])
        X_valid = np.concatenate([valid[i][0] for i in range(len(valid))])
        X_test = np.concatenate([test[i][0] for i in range(len(test))])
        y_train = np.concatenate([train[i][1] for i in range(len(train))])
        y_valid = np.concatenate([valid[i][1] for i in range(len(valid))])
        y_test = np.concatenate([test[i][1] for i in range(len(test))])

        X_train = X_train.reshape((X_train.shape[0], np.prod(X_train.shape[1:])))
        X_valid = X_valid.reshape((X_valid.shape[0], np.prod(X_valid.shape[1:])))
        X_test = X_test.reshape((X_test.shape[0], np.prod(X_test.shape[1:])))

        return X_train, X_valid, X_test, y_train, y_valid, y_test

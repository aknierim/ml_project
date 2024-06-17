import shutil
from pathlib import Path

import click
import requests
import tomli
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


@click.command()
@click.option(
    "--output-directory",
    "-o",
    type=click.Path(exists=False, dir_okay=True),
    help="Output directory for the dataset.",
)
@click.option(
    "--url",
    "-u",
    type=str,
    help="URL leading to the dataset.",
    default=None,
)
@click.option(
    "--from-name",
    "-n",
    is_flag=True,
    help="Downloads the Radio Galaxy Dataset from a list of URLs.",
    default=False,
)
def main(output_directory, url, from_name):
    if not isinstance(output_directory, Path):
        output_directory = Path(output_directory)

    if output_directory.is_file():
        raise ValueError("Output directory is not a directory and already exists")

    if not output_directory.is_dir():
        output_directory.mkdir(parents=True)

    data = GetData(output_directory)

    if url:
        data.from_url(url)

    elif from_name:
        data.from_name()

    else:
        raise ValueError(
            "Please either provide a valid URL or use the --from-name flag"
            " to download the Radio Galaxy Dataset."
        )


if __name__ == "__main__":
    main()

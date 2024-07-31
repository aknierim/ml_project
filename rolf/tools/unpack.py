"""CLI tool that unpacks zip files."""

from pathlib import Path
from zipfile import ZipFile

import click


def unzip(source_path: str | Path, dest_path: str | Path) -> None:
    """Unpacks a zip file.

    Parameters
    ----------
    source_path : str or Path
        Path to the input zip file.
    dest_path : str or Path
        Destination path.
    """
    source_path = Path(source_path)
    dest_path = Path(dest_path)

    with ZipFile(source_path, "r") as file:
        file.extractall(dest_path)


@click.command()
@click.argument(
    "input", type=click.Path(exists=True, dir_okay=False, file_okay=True), nargs=1
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=True, file_okay=False),
    help="Output path to unpack to.",
)
def main(input, output):
    unzip(source_path=input, dest_path=output)


if __name__ == "__main__":
    main()

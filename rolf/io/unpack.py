from pathlib import Path
from zipfile import ZipFile

import click


def unzip(source_path, dest_path) -> None:
    source_path = Path(source_path)
    dest_path = Path(dest_path)
    with ZipFile(source_path, "r") as file:
        file.extractall(dest_path)


@click.command()
@click.argument(
    "input",
    type=click.Path(exists=True, dir_okay=False, file_okay=True),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=True, file_okay=False),
    help="Output path to unpack to.",
)
def main(input, output):
    unzip(input, output)


if __name__ == "__main__":
    main()

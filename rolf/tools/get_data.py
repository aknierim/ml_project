"""CLI tool for data downloads."""

from pathlib import Path

import click

from rolf.io import GetData


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
def main(output_directory: str | Path, url: str, from_name: bool):
    """CLI tool to download data.

    Parameters
    ----------
    output_directory : str or Path
        Output directory for the downloaded data.
    url : str
        URL to download files from.
    from_name : bool
        If `True`, download from list of URLs found
        in urls.toml in the root of the repository.
    """
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

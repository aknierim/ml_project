import os
import time
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.pretty import Pretty
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from rolf.tools import ReadConfig


class ConsolePanel(Console):
    def __init__(self, *args, **kwargs):
        console_file = open(os.devnull, "w")
        super().__init__(record=True, file=console_file, *args, **kwargs)

    def __rich_console__(self, console, options):
        texts = self.export_text(clear=False).split("\n")
        for line in texts[-options.height :]:
            yield line.replace("'", "")


class Interface:
    def __init__(self, layout) -> None:
        self.layout = layout
        self.console: list[ConsolePanel] = [ConsolePanel() for _ in range(2)]

    def get_renderable(self):
        self.layout["body"].update(
            Panel(self.console[0], border_style="yellow", title="Training Information")
        )
        self.layout.children[0]
        return self.layout


def set_layout() -> Layout:
    """Define the layout."""
    layout = Layout(name="root")

    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=7),
    )
    layout["main"].split_row(
        Layout(name="side"),
        Layout(name="body", ratio=2, minimum_size=60),
    )
    layout["side"].split(Layout(name="box1"), Layout(name="box2"))
    return layout


def config_layout():
    conf_layout = Layout()
    conf_layout.split(Layout(name="upper"), Layout(name="lower", ratio=5))
    conf_layout["lower"].split_row(Layout(name="left"), Layout(name="right", ratio=4))

    return conf_layout


class Header:
    """Display header with clock."""

    def __rich__(self) -> Panel:
        grid = Table.grid(expand=True)
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="right")
        grid.add_row(
            "[b]ROLF[/b] Training",
            datetime.now().ctime().replace(":", "[blink]:[/]"),
        )
        return Panel(grid, style="white on blue")


class Training:
    def __init__(self, config_path):
        """ """
        if not isinstance(config_path, Path):
            config_path = Path(config_path)

        self.config_path = config_path

    def get_config(self, cli=None):
        """ """
        self.config = ReadConfig(self.config_path).training()

        if not cli:
            return self.config
        else:
            gl = Table.grid()
            gr = Table.grid()
            gl.add_column(justify="right")
            gr.add_column(justify="left")

            for key, vals in self.config.items():
                gl.add_row(f"[bold magenta]{key.upper()}")
                for k, v in vals.items():
                    gl.add_row("")
                    gr.add_row(f"{k}: {v}")
                gr.add_row("")

        return self.config, gl, gr

    def start(self, layout=None, progress=None):
        if layout is not None:
            db = Interface(layout)

            with Live(get_renderable=db.get_renderable):
                faux_training(
                    self.config["parameters"]["epochs"],
                    progress=progress,
                    console=db.console[0],
                )

        else:
            faux_training(self.config["parameters"]["epochs"], progress=progress)


def faux_training(epochs, progress=None, console=None):
    if console is not None:
        for i in range(epochs):
            console.print(
                Pretty(
                    "[yellow][INFO][reset]: Training epoch "
                    f"[steel_blue1]{i:3.0f}[reset] "
                    "--- Loss: "
                    f"[medium_spring_green]{1/(i + 1)**(1/2):.4f}[reset]"
                )
            )
            time.sleep(0.1)
            if progress and not progress.finished:
                progress.advance(progress.tasks[0].id)


def progress() -> Progress:
    job_progress = Progress(
        "{task.description}",
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("| Elapsed Time "),
        TimeElapsedColumn(),
        TextColumn("| Remaining Time"),
        TimeRemainingColumn(),
    )
    return job_progress


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Path to the config file.",
)
def main(config):
    config_path = Path(config)

    tr = Training(config)
    config, conf_left, conf_right = tr.get_config(cli=True)

    conf_layout = config_layout()
    conf_layout["upper"].update(f"File: {config_path.absolute()}")
    conf_layout["left"].update(conf_left)
    conf_layout["right"].update(conf_right)

    training_progress = progress()
    training_progress.add_task("[green]Training", total=config["parameters"]["epochs"])

    progress_table = Table.grid(expand=True)
    progress_table.add_row(
        Panel(
            training_progress, title="[b]Progress", border_style="blue", padding=(1, 2)
        ),
    )

    layout = set_layout()
    layout["header"].update(Header())
    # layout["box2"].update(Panel(make_syntax(), border_style="green"))
    layout["box1"].update(Panel(conf_layout, border_style="blue", title="Config"))
    layout["footer"].update(progress_table)

    tr.start(layout, training_progress)


if __name__ == "__main__":
    main()

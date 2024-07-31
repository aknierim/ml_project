"""GUI-based image viewer."""

import sys
from pathlib import Path

import click
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ImageViewerApp(QMainWindow):
    """Image viewer class."""

    def __init__(self, directory: str | Path, recursive: bool):
        """Initializes the image viewer.

        Parameters
        ----------
        directory : str or Path
            Path to the image directory.
        recursive : bool
            If `True`, also look for images in sub-directories.
        """
        super().__init__()

        if not directory:
            raise ValueError("Pleaser provide a directory with images!")

        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError("Please provide a path to a directory!")

        if recursive:
            self.image_paths = list(directory.glob("**/*[.png .jpeg .jpg]"))
        else:
            self.image_paths = list(directory.glob("*[.png .jpeg .jpg]"))

        if not self.image_paths:
            raise FileNotFoundError(
                f"No image files found in {directory.absolute()}!"
                " Make sure there are .jpg or .png files under"
                " the specified path."
            )

        self.setWindowTitle("Image Viewer")
        self.setGeometry(75, 75, 720, 405)

        qwidget = QWidget()
        self.setCentralWidget(qwidget)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.text_label = QLabel()
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        previous_button = QPushButton("Previous")
        next_button = QPushButton("Next")

        self.index = 0
        self.load_image()

        previous_button.clicked.connect(self.previous_image)
        next_button.clicked.connect(self.next_image)

        layout = QVBoxLayout()
        layout.addWidget(self.text_label)
        layout.addWidget(self.image_label)

        hor_layout = QHBoxLayout()
        hor_layout.addWidget(previous_button)
        hor_layout.addWidget(next_button)

        layout.addLayout(hor_layout)
        qwidget.setLayout(layout)

    def load_image(self) -> None:
        """Loads and displays the image at the current index."""
        n_images = len(self.image_paths) - 1

        if self.index < 0:
            self.index = n_images

        elif self.index > n_images:
            self.index = 0

        if 0 <= self.index <= n_images:
            image_path = self.image_paths[self.index]

            pixmap = QPixmap(image_path)
            pixmap = pixmap.scaledToWidth(500)

            self.image_label.setPixmap(pixmap)

            img_title = f"{image_path.name} | ({self.index}/{n_images})"
            self.text_label.setText(img_title)

    def previous_image(self) -> None:
        """Shows the previous image by decreasing the current
        index by one.
        """
        self.index -= 1
        self.load_image()

    def next_image(self) -> None:
        """Shows the next image by increasing the current
        index by one.
        """
        self.index += 1
        self.load_image()


@click.command()
@click.option(
    "--directory",
    "-d",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    help="Path to a directory containing <.jpg,.png> images.",
)
@click.option(
    "--recursive",
    "-r",
    is_flag=True,
    help="If True, also looks for images in sub-directories.",
)
def main(directory, recursive):
    app = QApplication(sys.argv)
    window = ImageViewerApp(directory, recursive)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

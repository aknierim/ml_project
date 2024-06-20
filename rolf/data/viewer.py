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
    def __init__(self, directory):
        super().__init__()

        if not directory:
            raise ValueError("Pleaser provide a directory with images!")

        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError("Please provide a path to a directory!")

        exts = [".png", ".jpeg", ".jpg"]
        self.image_paths = [p for p in directory.iterdir() if p.suffix in exts]

        if not self.image_paths:
            raise RuntimeError(
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

        # Connect button clicks to navigation methods
        previous_button.clicked.connect(self.previous_image)
        next_button.clicked.connect(self.next_image)

        # Create main layout
        layout = QVBoxLayout()
        layout.addWidget(self.text_label)
        layout.addWidget(self.image_label)

        # Create horizontal layout below image
        hor_layout = QHBoxLayout()
        hor_layout.addWidget(previous_button)
        hor_layout.addWidget(next_button)

        # Add horizontal layout to main layout
        # and set the widgets layout
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
def main(directory):
    app = QApplication(sys.argv)
    window = ImageViewerApp(directory)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

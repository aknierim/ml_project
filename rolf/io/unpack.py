from pathlib import Path
from zipfile import ZipFile 

def unzip(source_path, dest_path) -> None:
    source_path = Path(source_path)
    dest_path = Path(dest_path)
    with ZipFile(source_path, 'r') as file:
        file.extractall(dest_path)
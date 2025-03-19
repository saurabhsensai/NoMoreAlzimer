import tempfile
import shutil
import os
from pathlib import Path

class FileHandler:
    def __init__(self, config: dict):
        self.temp_dir = Path(config['paths']['temp_dir'])
        self.output_dir = Path(config['paths']['output_dir'])
        
    def create_temp_dir(self) -> Path:
        temp_dir = self.temp_dir / next(tempfile._get_candidate_names())
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir

    def save_uploaded_file(self, file_obj, dest_dir: Path) -> Path:
        dest_path = dest_dir / file_obj.name
        with open(dest_path, "wb") as f:
            f.write(file_obj.getbuffer())
        return dest_path

    def cleanup(self, path: Path):
        if path.exists():
            shutil.rmtree(path)

            
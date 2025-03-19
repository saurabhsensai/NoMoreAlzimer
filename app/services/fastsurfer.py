import subprocess
from typing import Generator
from app.models.schemas import SegmentationResult
from pathlib import Path
import os

class FastSurferService:
    def __init__(self, config: dict):
        # self.fs_home = Path(config['fastsurfer']['fs_home'])
        self.license_path = Path(config['fastsurfer']['license_path'])
        
    def segment(self, input_path: Path, output_dir: Path) -> Generator[str, None, SegmentationResult]:
        img_input_path = "/home/saurabh/core_ai/Final Application/" + os.path.dirname(input_path)
        actual_image = os.path.basename(input_path)
        
        print(img_input_path)
        print(actual_image)
        print(output_dir)

        # cmd = [
        #     "python", self.fs_home/"FastSurferCNN",
        #     "--t1", str(input_path),
        #     "--sid", "output",
        #     "--sd", str(output_dir),
        #     "--seg_only"
        # ]

        cmd = [
    "docker", "run", "--rm",
    "--gpus", "all",
    "--user", f"{subprocess.getoutput('id -u')}:{subprocess.getoutput('id -g')}",
    "-v", f"{img_input_path}:/data",  # Dynamically set input path
    "-v", f"/home/saurabh/core_ai/Final Application/{output_dir}:/output",  # Dynamically set output path
    "-v", "/home/saurabh/freesurfer_license:/fs_license",
    "deepmi/fastsurfer",
    "--fs_license", "/fs_license/license.txt",
    "--t1", f"/data/{actual_image}",  # Dynamically set actual image name
    "--sid", "subjectX",
    "--sd", "/output",
    "--seg_only",
    "--device", "cuda",
    "--no_cereb",
    "--no_hypothal",
    "--threads", "20"
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            while True:
                output = process.stdout.readline()
                if not output and process.poll() is not None:
                    break
                yield output.strip()
                
            return self._validate_output(output_dir)
            
        except Exception as e:
            return SegmentationResult(success=False, error=str(e))

    def _validate_output(self, output_dir: Path) -> SegmentationResult:
        orig_nu = output_dir /"subjectX"/ "mri" / "orig_nu.mgz"
        aseg = output_dir /"subjectX" / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
        
        if not orig_nu.exists() or not aseg.exists():
            return SegmentationResult(
                success=False,
                error="Missing output files"
            )
            
        return SegmentationResult(
            orig_nu_path=str(orig_nu),
            aseg_path=str(aseg),
            success=True
        )
from pydantic import BaseModel
from typing import Optional, List, Dict

class ProcessingConfig(BaseModel):
    input_path: str
    output_dir: str
    voxel_size: float = 1.0
    cleanup: bool = False

class SegmentationResult(BaseModel):
    orig_nu_path: Optional[str] = None
    aseg_path: Optional[str] = None
    success: bool = False
    error: Optional[str] = None
    metrics: Dict[str, float] = {}
import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

class Preprocessor:
    def __init__(self):
        self.default_voxel_size = 1.0

    def resample_to_isotropic(self, input_path: str, output_dir: str) -> str:
        """Resample MRI scan to isotropic voxels"""
        voxel_size = self.default_voxel_size
        img = nib.load(input_path)
        data = img.get_fdata()
        
        current_voxel_size = np.array([
            np.linalg.norm(img.affine[:3, i]) for i in range(3)
        ])
        
        zoom_factors = current_voxel_size / voxel_size
        resampled_data = zoom(data, zoom_factors, order=3)
        
        # Save resampled data
        output_path = os.path.join(output_dir, "preprocessed.nii")
        nib.save(nib.Nifti1Image(resampled_data, img.affine), output_path)
        return output_path
    


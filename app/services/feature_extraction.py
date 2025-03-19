import radiomics
from radiomics import featureextractor
import nibabel as nib
import pandas as pd

# Define important features to display
IMPORTANT_FEATURES = [
    "original_firstorder_Mean", "original_firstorder_Median",
    "original_glcm_Contrast", "original_glcm_Correlation",
    "original_gldm_DependenceEntropy", "original_glszm_GrayLevelVariance",
    "original_ngtdm_Coarseness", "original_shape_Elongation"
]

def extract_features(mri_path, mask_path, label):
    """
    Extracts and filters important radiomic features.

    Args:
        mri_path (str): Path to the MRI image (.mgz).
        mask_path (str): Path to the segmentation mask (.mgz).

    Returns:
        dict: Extracted and filtered features.
    """
    try:
        # Convert .mgz to .nii.gz for compatibility
        def convert_mgz_to_nii(mgz_path):
            img = nib.load(mgz_path)
            nii_path = mgz_path.replace('.mgz', '.nii.gz')
            nib.save(img, nii_path)
            return nii_path

        nii_mri_path = convert_mgz_to_nii(mri_path)
        nii_mask_path = convert_mgz_to_nii(mask_path)

        # Configure PyRadiomics extractor
        extractor = featureextractor.RadiomicsFeatureExtractor()
          # Extract all available features

        # Extract features
        feature_dict = extractor.execute(nii_mri_path, nii_mask_path, label=int(label))

        # Filter only important features
        filtered_features = {k: v for k, v in feature_dict.items() if k in IMPORTANT_FEATURES}

        return filtered_features

    except Exception as e:
        print(f"Error extracting features: {e}")
        return {}

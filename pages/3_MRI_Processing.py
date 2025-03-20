# app.py
import streamlit as st
import nibabel as nib
import numpy as np
from pathlib import Path
from app.controllers.processing import ProcessingController
from app.models.schemas import ProcessingConfig, SegmentationResult
from app.utils.config import load_config
import plotly.graph_objects as go
from skimage import measure
import matplotlib.pyplot as plt
import io
from app.services.feature_extraction import extract_features
import plotly.express as px
import pandas as pd
import base64

def set_bg_hack(image_path):
    try:
        bg_img = f"""
        .hero-section {{
            background-image: url("https://images.pexels.com/photos/4226264/pexels-photo-4226264.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1");
           
        }}
        """
    except Exception:
        bg_img = """
        .hero-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        """
    return bg_img

hero_bg = set_bg_hack('images/mri.png')

# Custom CSS with modern styling and hero background
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: #333333;
    }}
            
    .hero-section {{
        position: relative;
        text-align: center;
        padding: 4rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        min-height: 400px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    
    {hero_bg}
    .hero-overlay {{
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(255, 255, 255, 0);
        backdrop-filter: blur(0px);
    }}
    
    .hero-content {{
        position: relative;
        z-index: 2;
        max-width: 800px;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.90);
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }}
    
    .hero-title {{
        color: #000000 !important;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: none !important;
    }}
    
    .hero-subtitle {{
        color: #475569;
        font-size: 1.25rem;
        line-height: 1.6;
    }}

    .main {{
        padding: 2rem;
        background-color: #f8fafc;
    }}
    
    .stButton>button {{
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
        transition: all 0.3s ease;
        color: #ffffff;
    }}

    .data-form {{
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
    }}

    .delete-button {{
        background-color: #dc3545;
        color: white;
    }}

    .delete-button:hover {{
        background-color: #c82333;
        color: white;
    }}

    /* Statistics Cards Styling */
    .stat-card {{
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.3s ease;
        margin: 0.5rem 0;
    }}

    .stat-card:hover {{
        transform: translateY(-5px);
    }}

    .stat-card h3 {{
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }}

    .stat-card h2 {{
        font-size: 2rem;
        font-weight: 600;
        margin: 0;
    }}
    </style>
""", unsafe_allow_html=True)

 

# Hero Section
st.markdown("""
    <div class="hero-section">
        <div class="hero-overlay"></div>
        <div class="hero-content">
            <h1 class="hero-title">Brain feature extraction and visualization</h1>
            <p class="hero-subtitle">
                Advanced Data Management for Alzheimer's Disease Research<br>
                Streamlining patient data collection and analysis for breakthrough discoveries
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
        .stApp {
            max-width: 100%;
            padding: 0px;
            margin: 0px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Custom CSS Styles ---
st.markdown("""
<style>
    :root {
        --primary: #2A5C82;
        --secondary: #5DA9E9;
        --background: #F5F9FF;
        --card-bg: #FFFFFF;
    }

    .main {
        background-color: var(--background);
    }
    
    .stApp {
    
        padding: 1rem;
    }

    /* Improved upload container */
    .upload-container {
        border: 2px dashed var(--primary);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        background: var(--card-bg);
        transition: all 0.3s ease;
    }
    
    .upload-container:hover {
        border-color: var(--secondary);
        background: #f8fbff;
    }

    /* Modern header styling */
    .header {
        color: var(--primary);
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        text-align: center;
        margin: 2rem 0;
        position: relative;
    }
    
    .header:after {
        content: '';
        display: block;
        width: 60px;
        height: 4px;
        background: var(--secondary);
        margin: 0.5rem auto;
        border-radius: 2px;
    }

    /* Enhanced sidebar styling */
    [data-testid="stSidebar"] {
        background: var(--card-bg) !important;
        border-right: 1px solid #e6ecf0;
        padding: 1rem;
    }
    
    .sidebar-header {
        font-size: 1.2rem;
        color: var(--primary);
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Card styling for sections */
    .card {
        background: var(--card-bg);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e6ecf0;
    }
    
    .card-header {
        font-size: 1.1rem;
        color: var(--primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Improved button styling */
    .stButton>button {
        background: var(--primary) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 8px rgba(42,92,130,0.2);
    }

    /* Better slider styling */
    .stSlider .thumb {
        border-color: var(--primary) !important;
    }
    
    .stSlider .mark {
        color: var(--primary) !important;
    }

    /* Feature table styling */
    .feature-table {
        background: var(--card-bg) !important;
        border-radius: 8px !important;
        border: 1px solid #e6ecf0 !important;
    }
    
    .feature-table thead th {
        background: var(--background) !important;
        color: var(--primary) !important;
    }

    /* Badge styling for regions */
    .badge {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        background: var(--background);
        color: var(--primary);
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        border: 1px solid #dde7f0;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'mri_data' not in st.session_state:
    st.session_state.mri_data = None
if 'processing' not in st.session_state:
    st.session_state.processing = {
        'in_progress': False,
        'result': None,
        'uploaded_path': None
    }
if 'select_all' not in st.session_state:
    st.session_state.select_all = False
if 'previous_selected_regions' not in st.session_state:
    st.session_state.previous_selected_regions = []
if 'show_mesh' not in st.session_state:
    st.session_state.show_mesh = True 
if 'use_example' not in st.session_state:
    st.session_state.use_example = False
if 'selected_subject' not in st.session_state:
    st.session_state.selected_subject = None
if 'rotation_angle' not in st.session_state:
    st.session_state.rotation_angle = 0  # 0 degrees (no rotation)

def load_example_data(subject_id):
    """Load example data for the selected subject"""
    demo_dir = Path("demo")
    subject_dir = demo_dir / subject_id
    
    # Check if subject directory exists
    if not subject_dir.exists():
        st.error(f"Subject directory not found: {subject_dir}")
        return None
    
    # Paths to MRI and segmentation files
    orig_path = subject_dir / "orig_nu.mgz"
    aseg_path = subject_dir / "aparc.DKTatlas+aseg.deep.mgz"
    
    # Check if files exist
    if not orig_path.exists() or not aseg_path.exists():
        st.error(f"Required files not found for subject {subject_id}")
        return None
    
    # Create a result object similar to what the processing would return
    result = SegmentationResult(
        success=True,
        message="Example data loaded successfully",
        orig_nu_path=str(orig_path),
        aseg_path=str(aseg_path)
    )
    
    # Load segmentation data
    try:
        aseg_img = nib.load(aseg_path)
        mri_data = aseg_img.get_fdata().astype(np.int16)
        return result, mri_data
    except Exception as e:
        st.error(f"Error loading example data: {str(e)}")
        return None, None



def get_slice(volume, orientation, slice_idx, rotation_angle=0):
    """Extract a slice from the volume based on orientation and apply rotation"""
    if orientation == "axial":
        slice_data = volume[..., slice_idx]
    elif orientation == "coronal":
        slice_data = volume[:, slice_idx, :]
    elif orientation == "sagittal":
        slice_data = volume[slice_idx, :, :]
    
    # Apply rotation
    k = (rotation_angle // 90) % 4  
    if k > 0:
        slice_data = np.rot90(slice_data, k=k)
    
    return slice_data
    

def create_slice_visualization(orig_img_path, seg_img_path, orientation="axial", slice_idx=None, rotation_angle=0):
    """Create a visualization of the original MRI and segmentation for a given slice with rotation"""
    try:
        # Load the MRI and segmentation data
        orig_img = nib.load(orig_img_path)
        seg_img = nib.load(seg_img_path)
        
        # Get the data as numpy arrays
        orig_data = orig_img.get_fdata()
        seg_data = seg_img.get_fdata().astype(np.int16)
        
        # Determine the dimensions based on orientation
        if orientation == "axial":
            max_slice = orig_data.shape[2] - 1
        elif orientation == "coronal":
            max_slice = orig_data.shape[1] - 1
        elif orientation == "sagittal":
            max_slice = orig_data.shape[0] - 1
        
        # If slice_idx is None, default to the middle slice
        if slice_idx is None:
            slice_idx = max_slice // 2
        
        # Ensure slice_idx is within bounds
        slice_idx = max(0, min(slice_idx, max_slice))
        
        # Get the slices for both original and segmentation data with rotation
        orig_slice = get_slice(orig_data, orientation, slice_idx, rotation_angle)
        seg_slice = get_slice(seg_data, orientation, slice_idx, rotation_angle)
        
        # Normalize the original image for display
        orig_slice_norm = (orig_slice - orig_slice.min()) / (orig_slice.max() - orig_slice.min())
        
        # Create the figure with two subplots
        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display the original MRI slice
        ax1.imshow(np.rot90(orig_slice_norm), cmap="gray")
        ax1.set_title("Original MRI")
        ax1.axis("off")
        
        # Use skimage.color.label2rgb to colorize the segmentation mask
        from skimage import color
        seg_slice_rotated = np.rot90(seg_slice)
        color_seg = color.label2rgb(seg_slice_rotated, bg_label=0)
        
        # Display the overlay
        ax3.imshow(np.rot90(orig_slice_norm), cmap="gray")
        ax3.imshow(color_seg, alpha=0.5)
        ax3.set_title("Overlay")
        ax3.axis("off")
        
        # Set the figure title
        fig.suptitle(f"{orientation.capitalize()} View - Slice {slice_idx}/{max_slice} (Rotation: {rotation_angle}°)", fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        plt.close(fig)
        
        return buf, max_slice
    
    except Exception as e:
        st.error(f"Error creating slice visualization: {str(e)}")
        return None, 0

# Function to handle select all toggle
def toggle_select_all():
    st.session_state.select_all = not st.session_state.select_all
    
    # Get all region options
    labels_dict = initialize_labels()
    region_options = list(labels_dict.keys())
    
    # Update selected regions based on select_all state
    if st.session_state.select_all:
        st.session_state.previous_selected_regions = region_options
    else:
        st.session_state.previous_selected_regions = []

        
# --- Configuration and Processing Setup ---
config = load_config()
processing_controller = ProcessingController(config)
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# --- Core Functions ---
def save_uploaded_file(uploaded_file):
    """Save uploaded file to persistent storage"""
    file_path = UPLOAD_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def initialize_labels():
    """FreeSurfer segmentation labels"""
    return {
        'Lateral-Ventricle': {'left': 4, 'right': 43},
        'Inf-Lat-Vent': {'left': 5, 'right': 44},
        'Cerebellum-White-Matter': {'left': 7, 'right': 46},
        'Cerebellum-Cortex': {'left': 8, 'right': 47},
        'Thalamus-Proper': {'left': 10, 'right': 49},
        'Caudate': {'left': 11, 'right': 50},
        'Putamen': {'left': 12, 'right': 51},
        'Pallidum': {'left': 13, 'right': 52},
        '3rd-Ventricle': {'left': 14, 'right': 14},  # Same for both hemispheres
        '4th-Ventricle': {'left': 15, 'right': 15},  # Same for both hemispheres
        'Brain-Stem': {'left': 16, 'right': 16},     # Same for both hemispheres
        'Hippocampus': {'left': 17, 'right': 53},
        'Amygdala': {'left': 18, 'right': 54},
        'CSF': {'left': 24, 'right': 24},           # Same for both hemispheres
        'Accumbens-area': {'left': 26, 'right': 58},
        'VentralDC': {'left': 28, 'right': 60},
        'choroid-plexus': {'left': 31, 'right': 63},
        
        # Cortical regions - left hemisphere uses 1000+, right hemisphere uses 2000+
        'caudalanteriorcingulate': {'left': 1002, 'right': 2002},
        'caudalmiddlefrontal': {'left': 1003, 'right': 2003},
        'cuneus': {'left': 1005, 'right': 2005},
        'entorhinal': {'left': 1006, 'right': 2006},
        'fusiform': {'left': 1007, 'right': 2007},
        'inferiorparietal': {'left': 1008, 'right': 2008},
        'inferiortemporal': {'left': 1009, 'right': 2009},
        'isthmuscingulate': {'left': 1010, 'right': 2010},
        'lateraloccipital': {'left': 1011, 'right': 2011},
        'lateralorbitofrontal': {'left': 1012, 'right': 2012},
        'lingual': {'left': 1013, 'right': 2013},
        'medialorbitofrontal': {'left': 1014, 'right': 2014},
        'middletemporal': {'left': 1015, 'right': 2015},
        'parahippocampal': {'left': 1016, 'right': 2016},
        'paracentral': {'left': 1017, 'right': 2017},
        'parsopercularis': {'left': 1018, 'right': 2018},
        'parsorbitalis': {'left': 1019, 'right': 2019},
        'parstriangularis': {'left': 1020, 'right': 2020},
        'pericalcarine': {'left': 1021, 'right': 2021},
        'postcentral': {'left': 1022, 'right': 2022},
        'posteriorcingulate': {'left': 1023, 'right': 2023},
        'precentral': {'left': 1024, 'right': 2024},
        'precuneus': {'left': 1025, 'right': 2025},
        'rostralanteriorcingulate': {'left': 1026, 'right': 2026},
        'rostralmiddlefrontal': {'left': 1027, 'right': 2027},
        'superiorfrontal': {'left': 1028, 'right': 2028},
        'superiorparietal': {'left': 1029, 'right': 2029},
        'superiortemporal': {'left': 1030, 'right': 2030},
        'supramarginal': {'left': 1031, 'right': 2031},
        'transversetemporal': {'left': 1034, 'right': 2034},
        'insula': {'left': 1035, 'right': 2035}
    }



def create_region_visualization(data, selected_regions, hemisphere='both', 
                                show_mesh=True, opacity=0.7, colorscale_name='Viridis'):
    if not selected_regions:
        return None
    
    labels_dict = initialize_labels()
    
    # Prepare the figure
    fig = go.Figure()
    
    # Color palette for different regions
    colors = [
        '#FF5733', '#33FF57', '#3357FF', '#F033FF', '#FF33F0',
        '#33FFF0', '#F0FF33', '#8C33FF', '#FF8C33', '#33FF8C',
        '#8CFF33', '#338CFF', '#FF338C', '#33FFF0', '#F033FF'
    ]
    
    # Process each selected region
    for i, region_info in enumerate(selected_regions):
        region_name, hemi = region_info
        
        # Skip if region not in our labels dictionary
        if region_name not in labels_dict:
            continue
        
        # Get label values based on hemisphere
        if hemi == 'both':
            label_values = [labels_dict[region_name]['left'], labels_dict[region_name]['right']]
        else:
            label_values = [labels_dict[region_name][hemi]]
        
        # Create binary mask for this region
        region_mask = np.zeros_like(data, dtype=bool)
        for label_val in label_values:
            region_mask |= (data == label_val)
        
        # Skip if no voxels found for this region
        if not np.any(region_mask):
            continue
        
        # Use marching cubes to get surface mesh
        try:
            verts, faces, normals, values = measure.marching_cubes(region_mask, 0)
            
            # Use color from our palette (cycling if we have more regions than colors)
            color = colors[i % len(colors)]
            
            # Add the surface mesh to the figure
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=opacity,
                color=color,
                name=f"{region_name} ({hemi})"
            ))
            
            # Optionally add wireframe mesh
            if show_mesh:
                tri_vertices = verts[faces]
                Xe = []
                Ye = []
                Ze = []
                
                for T in tri_vertices:
                    Xe += [T[k%3][0] for k in range(4)] + [None]
                    Ye += [T[k%3][1] for k in range(4)] + [None]
                    Ze += [T[k%3][2] for k in range(4)] + [None]
                   
                fig.add_trace(go.Scatter3d(
                    x=Xe,
                    y=Ye,
                    z=Ze,
                    mode='lines',
                    name='',
                    line=dict(color='rgb(70,70,70)', width=0.5),
                    showlegend=False
                ))
        
        except Exception as e:
            
            # Continue with next region if one fails
            continue
    
    # Check if we have any traces (some regions might have failed or not been found)
    if len(fig.data) == 0:
        return None
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(showticklabels=False, title=''),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# --- Main Application Layout ---
# --- Sidebar Controls ---
with st.sidebar:
     
    st.markdown("""
        <div class="sidebar-header">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-sliders"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line></svg>
            Analysis Controls
        </div>
    """, unsafe_allow_html=True)
    st.header("Processing Controls")
    use_example = st.checkbox("Use Example Data", value=st.session_state.use_example)
    st.session_state.use_example = use_example
    
    if use_example:
        # Get list of available subjects from demo directory
        demo_dir = Path("demo")
        if demo_dir.exists():
            subjects = [d.name for d in demo_dir.iterdir() if d.is_dir()]
            if subjects:
                selected_subject = st.selectbox(
                    "Select Subject", 
                    options=subjects,
                    key="subject_selector"
                )
                
                if st.button("Load Example Data"):
                    result, mri_data = load_example_data(selected_subject)
                    if result and mri_data is not None:
                        st.session_state.processing['result'] = result
                        st.session_state.mri_data = mri_data
                        st.success(f"Example data for subject {selected_subject} loaded successfully!")
            else:
                st.error("No example subjects found in the demo directory.")
        else:
            st.error("Demo directory not found. Please create a 'demo' folder with subject data.")
    else:
    
    # Upload Section
        uploaded_file = st.file_uploader("Upload MRI Scan (.nii)", type=["nii", "nii.gz"])
        
        if uploaded_file and not st.session_state.processing['in_progress']:
            if st.button("Start Processing"):
                try:
                    # Save file and update state
                    file_path = save_uploaded_file(uploaded_file)
                    st.session_state.processing.update({
                        'in_progress': True,
                        'uploaded_path': file_path,
                        'result': None
                    })
                    
                    # Setup processing configuration
                    processing_config = ProcessingConfig(
                        input_path=str(file_path),
                        output_dir=str(UPLOAD_DIR)
                    )
                    
                    # Process scan with progress
                    messages = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process messages
                    for i, message in enumerate(processing_controller.process_scan(processing_config)):
                        messages.append(message)
                        progress = int((i+1)/20*100)  # Adjust 20 to match your actual step count
                        progress_bar.progress(min(progress, 100))
                        
                        if isinstance(message, str):
                            status_text.text(f"Status: {message}")
                        elif isinstance(message, SegmentationResult):
                            st.session_state.processing['result'] = message
                    
                    # Handle results
                    if st.session_state.processing['result'] and st.session_state.processing['result'].success:
                        # Load segmentation data
                        aseg_img = nib.load(st.session_state.processing['result'].aseg_path)
                        st.session_state.mri_data = aseg_img.get_fdata().astype(np.int16)
                        st.success("Processing completed successfully!")
                    else:
                        st.error("Processing failed")
                    
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")
                finally:
                    st.session_state.processing['in_progress'] = False
                    progress_bar.empty()
                    status_text.empty()

    # Visualization Controls
    if st.session_state.mri_data is not None:
        st.header("Visualization Settings")
        
        # Get label options
        labels_dict = initialize_labels()
        region_options = list(labels_dict.keys())
        
        # Add "Select All" checkbox
        select_all = st.checkbox("Select All Regions", 
                                value=st.session_state.select_all,
                                on_change=toggle_select_all)
        
        # Multiple region selection
        selected_regions = st.multiselect(
            "Select Regions",
            options=region_options,
            default=st.session_state.previous_selected_regions if st.session_state.previous_selected_regions else ['Hippocampus', 'Amygdala']
        )
        
        # Keep select_all checkbox in sync with the multiselect
        if len(selected_regions) == len(region_options) and not st.session_state.select_all:
            st.session_state.select_all = True
        elif len(selected_regions) < len(region_options) and st.session_state.select_all:
            st.session_state.select_all = False
        
        # Store the selected regions for next render
        st.session_state.previous_selected_regions = selected_regions
        
        hemisphere = st.radio("Hemisphere", ['left', 'right', 'both'])
        show_mesh = st.checkbox("Show Wireframe", value=st.session_state.show_mesh)
        st.session_state.show_mesh = show_mesh 
        opacity = st.slider("Opacity", 0.1, 1.0, 0.7)

# --- Main Visualization Area ---
if st.session_state.mri_data is not None:
    col1, col2 = st.columns([1, 5])
    
    with col1:
        st.header("Selected Regions")
        st.write(f"Loaded data shape: {st.session_state.mri_data.shape}")
        st.write(f"Selected regions: {', '.join(selected_regions)}")
    
    with col2:
        selected_region_info = [(region, hemisphere) for region in selected_regions]
        fig = create_region_visualization(
            st.session_state.mri_data,
            selected_region_info,
            hemisphere=hemisphere,
            show_mesh=st.session_state.show_mesh,  # Use the value from session state
            opacity=opacity
        )
        st.plotly_chart(fig, use_container_width=True, height=800)
else:
    st.markdown("""
    <div style="
        height: 500px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        background-color: #f5f5f5;
        border-radius: 10px;
    ">
        <h2 style="color: #666;">No Data Loaded</h2>
        <p style="color: #888;">Upload and process an MRI scan to begin</p>
    </div>
    """, unsafe_allow_html=True)

if st.session_state.mri_data is not None and st.session_state.processing['result'] is not None:
    st.markdown("---")
    st.header("2D Slice Viewer")
    
    col1, col2, col3 = st.columns([1, 4, 1]) 
    
    with col1:
        orientation = st.selectbox(
            "View Orientation",
            options=["axial", "coronal", "sagittal"],
            key="slice_orientation"
        )
    
    with col3:
        # Add the rotation button
        if st.button("Rotate 90° CW", key="rotate_button"):
            # Increment rotation angle by 90 degrees on each click
            st.session_state.rotation_angle = (st.session_state.rotation_angle + 90) % 360

    st.caption(f"Current rotation: {st.session_state.rotation_angle}°")        
    # Initialize slice index in session state if not exists
    if 'slice_idx' not in st.session_state:
        st.session_state.slice_idx = None
    
    # Get paths from the processing result
    orig_img_path = st.session_state.processing['result'].orig_nu_path
    seg_img_path = st.session_state.processing['result'].aseg_path
    
    # Create the visualization and get the maximum slice index
    viz_buf, max_slice = create_slice_visualization(
        orig_img_path, 
        seg_img_path, 
        orientation=orientation,
        slice_idx=st.session_state.slice_idx,
        rotation_angle=st.session_state.rotation_angle  # Pass the rotation angle
    )
    
    with col2:
        # Initialize slice_idx to middle if it's None
        if st.session_state.slice_idx is None:
            st.session_state.slice_idx = max_slice // 2
            
        # Slice selector slider
        slice_idx = st.slider(
            "Slice Number", 
            0, max_slice, 
            st.session_state.slice_idx,
            key="slice_selector"
        )
        st.session_state.slice_idx = slice_idx
    
    # If orientation or slice changed, update the visualization
    if orientation != st.session_state.get('last_orientation', '') or \
       slice_idx != st.session_state.get('last_slice_idx', -1) or \
       st.session_state.rotation_angle != st.session_state.get('last_rotation_angle', 0):
        
        viz_buf, _ = create_slice_visualization(
            orig_img_path, 
            seg_img_path, 
            orientation=orientation,
            slice_idx=slice_idx,
            rotation_angle=st.session_state.rotation_angle  # Pass the rotation angle
        )
        
        # Update last values
        st.session_state.last_orientation = orientation
        st.session_state.last_slice_idx = slice_idx
        st.session_state.last_rotation_angle = st.session_state.rotation_angle
    
    # Display the visualization
    if viz_buf:
        st.image(viz_buf, use_container_width=True)
    else:
        st.error("Failed to generate slice visualization")



# --- UI Styling ---
st.markdown("""
    <style>
        .feature-card {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 10px;
        }
        .feature-title {
            font-size: 16px;
            font-weight: bold;
            color: #333;
        }
        .feature-value {
            font-size: 14px;
            color: #007BFF;
        }
    </style>
""", unsafe_allow_html=True)

# --- Feature Extraction Section ---
st.markdown("---")
st.header("🧠 Radiomic Feature Extraction")

# Retrieve MRI and mask paths from session state
if "processing" in st.session_state and st.session_state.processing.get("result"):
    mri_path = st.session_state.processing["result"].orig_nu_path
    mask_path = st.session_state.processing["result"].aseg_path

    # Get label options from initialize_labels
    labels_dict = initialize_labels()

    # Create options for dropdown with left/right hemisphere specification
    feature_options = []
    for region_name, hemispheres in labels_dict.items():
        if 'left' in hemispheres and 'right' in hemispheres:
            if hemispheres['left'] != hemispheres['right']:  # Different values for left and right
                feature_options.append(f"Left {region_name}")
                feature_options.append(f"Right {region_name}")
            else:  # Same value for both hemispheres
                feature_options.append(region_name)
    
    # Sort options alphabetically
    feature_options.sort()
    
    # Multi-select for features
    selected_features = st.multiselect(
        "Select regions for feature extraction:",
        options=feature_options,
        default=["Left Hippocampus", "Right Hippocampus"]
    )
    
    if st.button("Extract Features"):
        if not selected_features:
            st.warning("⚠️ Please select at least one region for feature extraction.")
        else:
            with st.spinner("Extracting features..."):
                # Dictionary to store results for all regions
                all_features = {}
                
                # Process each selected feature
                for feature in selected_features:
                    # Parse the selected feature to get region and hemisphere
                    if feature.startswith("Left "):
                        region_name = feature[5:]  # Remove "Left " prefix
                        hemisphere = "left"
                    elif feature.startswith("Right "):
                        region_name = feature[6:]  # Remove "Right " prefix
                        hemisphere = "right"
                    else:
                        region_name = feature
                        hemisphere = "left"  # For regions with same value in both hemispheres
                    
                    # Get the label value
                    if region_name in labels_dict:
                        label_value = labels_dict[region_name][hemisphere]
                        
                        # Extract features for this region
                        try:
                            region_features = extract_features(mri_path, mask_path, label_value)
                            if region_features:
                                all_features[feature] = region_features
                            else:
                                all_features[feature] = {"Status": "No features found"}
                        except Exception as e:
                            all_features[feature] = {"Status": f"Error: {str(e)}"}
                    else:
                        all_features[feature] = {"Status": "Region not found in mask"}
                
                # Display success message if any features were extracted
                if any(not isinstance(features.get("Status", None), str) for features in all_features.values()):
                    st.success("✅ Feature extraction completed!")
                
                # Convert to DataFrame for easier display and download
                df_rows = []
                for region, features in all_features.items():
                    if "Status" in features:
                        # This region had an error or wasn't found
                        row = {"Region": region, "Status": features["Status"]}
                        df_rows.append(row)
                    else:
                        # Add all features for this region
                        row = {"Region": region}
                        row.update({k.replace("original_", ""): round(float(v), 4) for k, v in features.items()})
                        df_rows.append(row)
                
                features_df = pd.DataFrame(df_rows)
                
                # Display the features
                st.subheader("📊 Extracted Features")
                
                if not features_df.empty:
                    with st.expander("📌 Click to View Features", expanded=True):
                        st.dataframe(features_df)
                    
                    # Create download button for CSV
                    csv = features_df.to_csv(index=False)
                    st.download_button(
                        label="Download Features as CSV",
                        data=csv,
                        file_name="brain_region_features.csv",
                        mime="text/csv",
                    )
                else:
                    st.error("❌ No features were extracted successfully.")
else:
    st.warning("⚠️ MRI and Mask paths are not available. Please process an MRI scan first.")
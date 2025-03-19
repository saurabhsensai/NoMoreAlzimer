import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import base64


from utils import  apply_custom_css, make_classification_prediction
def get_base64_of_bin_file(image_path):
    with open(image_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

#  background-image: url("data:image/png;base64,{bin_str}");

def set_bg_hack(image_path):
    try:
        bin_str = get_base64_of_bin_file(image_path)
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

# Page configuration
st.set_page_config(
    page_title="Alzheimer's Research Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Get background image CSS
hero_bg = set_bg_hack('images/mri.png')  # Replace with your image path

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
            <h1 class="hero-title">Alzheimer's Research Platform</h1>
            <p class="hero-subtitle">
                Advanced Data Management for Alzheimer's Disease Research<br>
                Streamlining patient data collection and analysis for breakthrough discoveries
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)
# Statistics Cards with enhanced styling
stats_container = st.container()
# Define required columns for CSV upload
REQUIRED_COLUMNS = ['RID',
    'C2N_plasma_ptau217_ratio', 'C2N_plasma_Abeta42', 'C2N_plasma_Abeta42_Abeta40',
    'C2N_plasma_ptau217', 'Roche_plasma_GFAP', 'C2N_plasma_nptau217', 'Roche_plasma_NfL',
    'AGE', 'MMSE', 'ADAS11', 'ADAS13', 'FAQ', 'APOE4', 'PTETHCAT', 'PTGENDER', 'PTEDUCAT'
]

# Initialize session state for form fields
session_state_keys = [
    'c2n_plasma_ptau217_ratio', 'c2n_plasma_abeta42', 'c2n_plasma_abeta42_abeta40',
    'c2n_plasma_ptau217', 'roche_plasma_gfap', 'c2n_plasma_nptau217', 'roche_plasma_nfl',
    'age', 'mmse', 'adas11', 'adas13', 'faq', 'apoe4', 'ptethcat', 'ptgender', 'ptedu'
]

for key in session_state_keys:
    if key not in st.session_state:
        st.session_state[key] = None

# Function to process uploaded file (only populates form fields)
def process_uploaded_file():
    if st.session_state.uploaded_file is not None:
        try:
            df = pd.read_csv(st.session_state.uploaded_file)
            
            # Validate CSV structure
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                st.error(f"Missing columns in CSV: {', '.join(missing_cols)}")
            else:
                # Populate form fields with the first row of data
                first_row = df.iloc[0]
                
                # Update session state for all input fields
                st.session_state.c2n_plasma_ptau217_ratio = first_row['C2N_plasma_ptau217_ratio']
                st.session_state.c2n_plasma_abeta42 = first_row['C2N_plasma_Abeta42']
                st.session_state.c2n_plasma_abeta42_abeta40 = first_row['C2N_plasma_Abeta42_Abeta40']
                st.session_state.c2n_plasma_ptau217 = first_row['C2N_plasma_ptau217']
                st.session_state.roche_plasma_gfap = first_row['Roche_plasma_GFAP']
                st.session_state.c2n_plasma_nptau217 = first_row['C2N_plasma_nptau217']
                st.session_state.roche_plasma_nfl = first_row['Roche_plasma_NfL']
                st.session_state.age = first_row['AGE']
                st.session_state.mmse = first_row['MMSE']
                st.session_state.adas11 = first_row['ADAS11']
                st.session_state.adas13 = first_row['ADAS13']
                st.session_state.faq = first_row['FAQ']
                st.session_state.apoe4 = first_row['APOE4']
                st.session_state.ptethcat = first_row['PTETHCAT']
                st.session_state.ptgender = first_row['PTGENDER']
                st.session_state.ptedu = first_row['PTEDUCAT']
        
        except Exception as e:
            st.error(f"Error processing CSV file: {str(e)}")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="uploaded_file", on_change=process_uploaded_file)

# Create two columns for layout
col1, col2 = st.columns(2)

# Input form for manual entry
with st.form("classification_form"):
    st.subheader("📋 Input Parameters")
    
    with col1:
        st.markdown("### Plasma Biomarkers")
        c2n_plasma_ptau217_ratio = st.number_input("C2N Plasma pTau217 Ratio", 
            value=st.session_state.get('c2n_plasma_ptau217_ratio', None), 
            min_value=0.0, key='c2n_plasma_ptau217_ratio')
        c2n_plasma_abeta42 = st.number_input("C2N Plasma Aβ42",
            value=st.session_state.get('c2n_plasma_abeta42', None), 
            min_value=0.0, key='c2n_plasma_abeta42')
        c2n_plasma_abeta42_abeta40 = st.number_input("C2N Plasma Aβ42/Aβ40 Ratio",
            value=st.session_state.get('c2n_plasma_abeta42_abeta40', None), 
            min_value=0.0, key='c2n_plasma_abeta42_abeta40')
        c2n_plasma_ptau217 = st.number_input("C2N Plasma pTau217",
            value=st.session_state.get('c2n_plasma_ptau217', None), 
            min_value=0.0, key='c2n_plasma_ptau217')
        roche_plasma_gfap = st.number_input("Roche Plasma GFAP", 
            value=st.session_state.get('roche_plasma_gfap', None), 
            min_value=0.0, key='roche_plasma_gfap')
        c2n_plasma_nptau217 = st.number_input("C2N Plasma npTau217",
            value=st.session_state.get('c2n_plasma_nptau217', None), 
            min_value=0.0, key='c2n_plasma_nptau217')
        roche_plasma_nfl = st.number_input("Roche Plasma NfL",
            value=st.session_state.get('roche_plasma_nfl', None), 
            min_value=0.0, key='roche_plasma_nfl')
        
    with col2:
        st.markdown("### Clinical Parameters")
        age = st.number_input("Age", 
            value=st.session_state.get('age', None), 
            min_value=0.0, max_value=120.0,  key='age')
        mmse = st.number_input("MMSE Score",
            value=st.session_state.get('mmse', None), 
            min_value=0.0, max_value=30.0, key='mmse')
        adas11 = st.number_input("ADAS11 Score",
            value=st.session_state.get('adas11', None), 
            min_value=0.0, key='adas11')
        adas13 = st.number_input("ADAS13 Score",
            value=st.session_state.get('adas13', None), 
            min_value=0.0, key='adas13')
        faq = st.number_input("FAQ Score",
            value=st.session_state.get('faq', None), 
            min_value=0.0, key='faq')
        apoe4 = st.selectbox("APOE4 Status", 
            options=[0, 1, 2],
            index=[0, 1, 2].index(st.session_state.get('apoe4', 0)) if st.session_state.get('apoe4') is not None else 0,
            key='apoe4')
        ptethcat = st.selectbox("Ethnicity Category", 
                    options=[0, 1, 2],
                    index=[0, 1, 2].index(st.session_state.get('ptethcat', 0)) 
                    if st.session_state.get('ptethcat') is not None else 0,
                    format_func=lambda x: {0: 'Not Hispanic/Latino', 1: 'Hispanic/Latino', 2: 'Unknown'}[x],
                    key='ptethcat')

        ptgender = st.selectbox("Gender", 
            options=[0, 1],
            index=[0, 1].index(st.session_state.get('ptgender', 0)) 
            if st.session_state.get('ptgender') is not None else 0,
            format_func=lambda x: {0: 'Female', 1: 'Male'}[x],
            key='ptgender')
        ptedu = st.number_input("Education in Years", 
            value=st.session_state.get('ptedu', None), 
            min_value=0, key='ptedu')

        
    submit_button = st.form_submit_button("Generate Classification")

# Handle submission
if submit_button:
    # Prepare input data
    input_data = {
        'RID': 1,
        'C2N_plasma_ptau217_ratio': c2n_plasma_ptau217_ratio,
        'C2N_plasma_Abeta42': c2n_plasma_abeta42,
        'C2N_plasma_Abeta42_Abeta40': c2n_plasma_abeta42_abeta40,
        'C2N_plasma_ptau217': c2n_plasma_ptau217,
        'Roche_plasma_GFAP': roche_plasma_gfap,
        'C2N_plasma_nptau217': c2n_plasma_nptau217,
        'Roche_plasma_NfL': roche_plasma_nfl,
        'AGE': age,
        'MMSE': mmse,
        'ADAS11': adas11,
        'ADAS13': adas13,
        'FAQ': faq,
        'APOE4': apoe4,
        'PTETHCAT': ptethcat,
        'PTGENDER': ptgender,
        'PTEDUCAT': ptedu
    }
    
    # Make prediction
    prediction = make_classification_prediction(input_data)
    
    if prediction is not None:
        st.markdown("### 🎯 Classification Results")
        
        # Define classes and colors
        class_names = ['CN', 'MCI', 'AD']
        colors = ['#2ecc71', '#f1c40f', '#e74c3c']
        
        # 1. Main Prediction Display
        max_prob_idx = np.argmax(prediction)
        max_prob_class = class_names[max_prob_idx]
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="prediction-box" style="text-align: center;">
                <h3 style='color: #0066cc;'>Primary Prediction</h3>
                <h1 style='color: {colors[max_prob_idx]}; font-size: 3em;'>{max_prob_class}</h1>
                 <h2 style='color: #004d99;'>{prediction[max_prob_idx]:.1%}</h2>
                <p style='color: #666;'>Confidence Level</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 2. Radar Chart
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=prediction * 100,
            theta=class_names,
            fill='toself',
            name='Prediction Probabilities'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            title="Probability Distribution (Radar View)"
        )
        
        # 3. Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction[max_prob_idx] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': colors[max_prob_idx]},
                'steps': [
                    {'range': [0, 33], 'color': "lightgray"},
                    {'range': [33, 66], 'color': "gray"},
                    {'range': [66, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            },
            title={'text': "Confidence Level"}
        ))
        
        # 4. Horizontal Bar Chart
        fig_bar = go.Figure(go.Bar(
            x=prediction * 100,
            y=class_names,
            orientation='h',
            marker_color=colors,
            text=[f'{p:.1f}%' for p in prediction * 100],
            textposition='auto',
        ))
        fig_bar.update_layout(
            title="Probability Distribution by Class",
            xaxis_title="Probability (%)",
            yaxis_title="Class",
            showlegend=False
        )
        
        # 5. Visualization Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Distribution", "🎯 Confidence", "📈 Details"])
        
        with tab1:
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_gauge, use_container_width=True)
            with col2:
                st.plotly_chart(fig_radar, use_container_width=True)
                
        with tab3:
            # Detailed probability table
            st.markdown("### Detailed Probability Breakdown")
            results_df = pd.DataFrame({
                'Class': class_names,
                'Probability': prediction,
                'Confidence Level': ['High' if p > 0.7 else 'Medium' if p > 0.4 else 'Low' for p in prediction]
            })
            styled_df = results_df.style.format({'Probability': '{:.2%}'}).background_gradient(subset=['Probability'], cmap='Blues')
            st.dataframe(styled_df, use_container_width=True)
            
            # Feature summary
            st.markdown("### Input Features Summary")
            features_df = pd.DataFrame([input_data])
            styled_features = features_df.style.background_gradient(cmap='viridis')
            st.dataframe(styled_features, use_container_width=True)
        
        # 6. Interpretation Guidelines
        st.markdown("### 📌 Interpretation Guidelines")
        st.info("""
        - **High Confidence**: > 70% probability
        - **Medium Confidence**: 40-70% probability
        - **Low Confidence**: < 40% probability
        
        Please note that these predictions should be considered alongside clinical judgment.
        """)

        # 7. Download Results
        results_dict = {
            'Prediction': max_prob_class,
            'Confidence': f"{prediction[max_prob_idx]:.1%}",
            **{f"{cls} Probability": f"{prob:.1%}" for cls, prob in zip(class_names, prediction)},
            **input_data
        }
        results_df = pd.DataFrame([results_dict])
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results",
            data=csv,
            file_name="classification_results.csv",
            mime="text/csv"
        )
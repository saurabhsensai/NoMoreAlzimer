import streamlit as st
import pandas as pd
from database import init_db, add_patient, get_all_patients, delete_patient
import datetime
import base64

# Initialize the database
init_db()

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
            background-image: url("https://images.pexels.com/photos/7089011/pexels-photo-7089011.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1");
           
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
        background: rgba(255, 255, 255, 0.80);
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


# Statistics Cards
col1, col2, col3, col4 = st.columns(4)
total_patients = get_all_patients()
total_patient_unique = len(total_patients['PTID'].unique())

with col1:
    st.markdown("""
        <div class="stat-card">
            <h3>📊 Total Patients</h3>
            <h2 style="color: #3b82f6;">{}</h2>
        </div>
    """.format(total_patient_unique), unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="stat-card">
            <h3>🏥 Active Studies</h3>
            <h2 style="color: #10b981;">3</h2>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="stat-card">
            <h3>🔬 Biomarkers</h3>
            <h2 style="color: #8b5cf6;">7</h2>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
        <div class="stat-card">
            <h3>📋 Assessments</h3>
            <h2 style="color: #f59e0b;">5</h2>
        </div>
    """, unsafe_allow_html=True)

# Create tabs with custom styling
tab1, tab2 = st.tabs(["📝 Data Entry", "📊 View Records"])

with tab1:
    st.markdown("""
        <div class="form-header">
            <h2>Patient Data Entry</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # CSV Upload Section with improved styling
    st.markdown("### 📤 Import Data from CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("🚀 Import CSV Data"):
                with st.status("Importing data...", expanded=True) as status:
                    success_count = 0
                    error_count = 0
                    
                    for _, row in df.iterrows():
                        data = {
                            'PTID': row.get('PTID'),
                            'NAME': row.get('NAME'),
                            'VISCODE': row.get('VISCODE'),
                            'C2N_plasma_ptau217_ratio': row.get('C2N_plasma_ptau217_ratio'),
                            'C2N_plasma_Abeta42': row.get('C2N_plasma_Abeta42'),
                            'C2N_plasma_Abeta42_Abeta40': row.get('C2N_plasma_Abeta42_Abeta40'),
                            'C2N_plasma_ptau217': row.get('C2N_plasma_ptau217'),
                            'Roche_plasma_GFAP': row.get('Roche_plasma_GFAP'),
                            'C2N_plasma_nptau217': row.get('C2N_plasma_nptau217'),
                            'Roche_plasma_NfL': row.get('Roche_plasma_NfL'),
                            'DX': row.get('DX'),
                            'AGE': row.get('AGE'),
                            'MMSE': row.get('MMSE'),
                            'ADAS11': row.get('ADAS11'),
                            'ADAS13': row.get('ADAS13'),
                            'CDRSB': row.get('CDRSB'),
                            'FAQ': row.get('FAQ'),
                            'APOE4': row.get('APOE4'),
                            'PTETHCAT': row.get('PTETHCAT'),
                            'PTEDUCAT': row.get('PTEDUCAT'),
                            'PTGENDER': row.get('PTGENDER')
                        }
                        
                        if not data['PTID'] or not data['NAME']:
                            error_count += 1
                            continue
                            
                        if add_patient(data):
                            success_count += 1
                        else:
                            error_count += 1
                    
                    status.update(label=f"✅ Imported {success_count} records successfully. ❌ {error_count} records failed.", 
                                state="complete")
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

    st.markdown("---")
    st.markdown("### ✍️ Manual Data Entry")
    
    with st.form("patient_data_form", clear_on_submit=True):
        # st.markdown('<div class="data-form">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 👤 Patient Information")
            ptid = st.text_input("Patient ID*", key="ptid")
            name = st.text_input("Patient Name*", key="name")
            viscode = st.selectbox("Visit Code*", ['bl', 'm06', 'm12', 'm18', 'm24', 'm30', 'm36', 'm42'], key="viscode")
            age = st.number_input("Age", min_value=0, max_value=120, key="age")
            ptgender = st.selectbox("Gender", ['Female', 'Male'], key="ptgender")
            ptethcat = st.selectbox("Ethnicity", ['Not Hispanic/Latino', 'Hispanic/Latino', 'Unknown'], key="ptethcat")
            ptedu = st.text_input("Education (Years)*", key="ptedu")
            dx = st.selectbox("Diagnosis*", ['CN', 'MCI', 'AD'], key="dx")
            apoe4 = st.selectbox("APOE4 Status*", [0, 1, 2], key="apoe4")

        with col2:
            st.markdown("#### 📊 Clinical Scores")
            mmse = st.number_input("MMSE Score", min_value=0.0, max_value=30.0, key="mmse")
            adas11 = st.number_input("ADAS11 Score", value=None, min_value=0.0, key="adas11")
            adas13 = st.number_input("ADAS13 Score", value=None, min_value=0.0, key="adas13")
            cdrsb = st.number_input("CDRSB Score", value=None, min_value=0.0, key="cdrsb")
            faq = st.number_input("FAQ Score", value=None, min_value=0.0, key="faq")

        with col3:
            st.markdown("#### 🧬 Plasma Biomarkers")
            c2n_plasma_ptau217_ratio = st.number_input("C2N Plasma pTau217 Ratio", value=None, min_value=0.0, key="c2n_plasma_ptau217_ratio")
            c2n_plasma_abeta42 = st.number_input("C2N Plasma Aβ42", value=None, min_value=0.0, key="c2n_plasma_abeta42")
            c2n_plasma_abeta42_abeta40 = st.number_input("C2N Plasma Aβ42/Aβ40 Ratio", value=None, min_value=0.0, key="c2n_plasma_abeta42_abeta40")
            c2n_plasma_ptau217 = st.number_input("C2N Plasma pTau217", value=None, min_value=0.0, key="c2n_plasma_ptau217")
            roche_plasma_gfap = st.number_input("Roche Plasma GFAP", value=None, min_value=0.0, key="roche_plasma_gfap")
            c2n_plasma_nptau217 = st.number_input("C2N Plasma npTau217", value=None, min_value=0.0, key="c2n_plasma_nptau217")
            roche_plasma_nfl = st.number_input("Roche Plasma NfL", value=None, min_value=0.0, key="roche_plasma_nfl")

        submit_button = st.form_submit_button("📥 Submit Patient Data")
        st.markdown('</div>', unsafe_allow_html=True)

        if submit_button:
            if not ptid or not name:
                st.error("❗ Patient ID and Name are required fields!")
            else:
                data = {
                    'PTID': ptid,
                    'NAME': name,
                    'VISCODE': viscode,
                    'C2N_plasma_ptau217_ratio': c2n_plasma_ptau217_ratio,
                    'C2N_plasma_Abeta42': c2n_plasma_abeta42,
                    'C2N_plasma_Abeta42_Abeta40': c2n_plasma_abeta42_abeta40,
                    'C2N_plasma_ptau217': c2n_plasma_ptau217,
                    'Roche_plasma_GFAP': roche_plasma_gfap,
                    'C2N_plasma_nptau217': c2n_plasma_nptau217,
                    'Roche_plasma_NfL': roche_plasma_nfl,
                    'DX': dx,
                    'AGE': age,
                    'MMSE': mmse,
                    'ADAS11': adas11,
                    'ADAS13': adas13,
                    'CDRSB': cdrsb,
                    'FAQ': faq,
                    'APOE4': apoe4,
                    'PTETHCAT': ptethcat,
                    'PTEDUCAT': ptedu,
                    'PTGENDER': ptgender
                }
                
                if add_patient(data):
                    st.success("✅ Patient data successfully added!")
                else:
                    st.error("❌ Error: Patient ID already exists!")

with tab2:
    st.markdown("""
        <div class="form-header">
            <h2>Patient Records Dashboard</h2>
        </div>
    """, unsafe_allow_html=True)
    
    search_col1, search_col2, delete_col = st.columns([2, 1, 1])
    with search_col1:
        search_term = st.text_input("🔍 Search by Patient ID or Name")
    with search_col2:
        sort_by = st.selectbox("📑 Sort by", ["PTID", "NAME", "AGE", "timestamp"])
    
    df = get_all_patients()
    
    if search_term:
        df = df[df['PTID'].str.contains(search_term, case=False) | 
                df['NAME'].str.contains(search_term, case=False)]
    
    df = df.sort_values(by=sort_by)
    df_display = df.copy()
    df_display['Select'] = False
    
    edited_df = st.data_editor(
        df_display,
        column_config={
            "Select": st.column_config.CheckboxColumn(
                "Delete",
                default=False
            )
        },
        disabled=df_display.columns.drop('Select').tolist(),
        use_container_width=True,
        height=400
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🗑️ Delete Selected Records", type="primary"):
            selected_ptids = edited_df[edited_df['Select']]['ID'].tolist()
            
            if not selected_ptids:
                st.warning("⚠️ No records selected!")
            else:
                with st.status("Processing deletions...", expanded=True) as status:
                    deleted_count = 0
                    for ptid in selected_ptids:
                        if delete_patient(ptid):
                            deleted_count += 1
                    status.update(label=f"✅ Deleted {deleted_count} records", state="complete")
                st.rerun()

    with col2:
        if not df.empty:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f"patient_data_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

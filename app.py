import streamlit as st
import pandas as pd
from database import init_db, add_patient, get_all_patients, delete_patient
import datetime


# Initialize the database
init_db()

# Page configuration
st.set_page_config(
    page_title="No More Alzheimers Disease",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .data-form {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
    }
    .delete-button {
        background-color: #dc3545;
        color: white;
    }
    .delete-button:hover {
        background-color: #c82333;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧬 No More Alzheimers Disease")

# Create tabs for different sections
tab1, tab2 = st.tabs(["📝 Data Entry", "📊 View Records"])

with tab1:
    st.header("Patient Data Entry Form")
    
    # Add CSV upload section
    st.subheader("Import Data from CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            if st.button("Import CSV Data"):
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
                        
                        # Check for required fields
                        if not data['PTID'] or not data['NAME']:
                            error_count += 1
                            continue
                            
                        if add_patient(data):
                            success_count += 1
                        else:
                            error_count += 1
                            
                    status.update(label=f"Imported {success_count} records successfully. {error_count} records failed.", 
                                state="complete")
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

    st.markdown("---")
    st.subheader("Manual Data Entry")
    
    with st.form("patient_data_form", clear_on_submit=True):
        # st.markdown('<div class="data-form">', unsafe_allow_html=True)
        
        # Create three columns for better layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Patient Information")
            ptid = st.text_input("Patient ID*", key="ptid")
            name = st.text_input("Patient Name*", key="name")
            viscode = st.selectbox("Visiting Code*", ['bl', 'm06', 'm12', 'm18', 'm24', 'm30' , 'm36' , 'm42'], key="viscode")
            age = st.number_input("Age", min_value=0, max_value=120, key="age")
            ptgender = st.selectbox("Gender", ['Female', 'Male'], key="ptgender")
            ptethcat = st.selectbox("Ethnicity Category", 
                                  ['Not Hispanic/Latino', 'Hispanic/Latino', 'Unknown'],
                                  key="ptethcat")
            ptedu = st.text_input("Education in Years*", key="ptedu")
            dx = st.selectbox("Diagnosis*", ['CN', 'MCI', 'AD'], key="dx")
            apoe4 = st.selectbox("APOE4 Status*", [0, 1, 2], key="apoe4")

        with col2:
            st.subheader("Clinical Scores")
            mmse = st.number_input("MMSE Score", min_value=0.0, max_value=30.0, key="mmse")
            adas11 = st.number_input("ADAS11 Score",value=None,  min_value=0.0, key="adas11")
            adas13 = st.number_input("ADAS13 Score", value=None, min_value=0.0, key="adas13")
            cdrsb = st.number_input("CDRSB Score",value=None,  min_value=0.0, key="cdrsb")
            faq = st.number_input("FAQ Score",value=None, min_value=0.0, key="faq")

        with col3:
            st.subheader("Plasma Biomarkers")
            c2n_plasma_ptau217_ratio = st.number_input("C2N Plasma pTau217 Ratio",value=None, min_value=0.0, 
                                                      key="c2n_plasma_ptau217_ratio")
            c2n_plasma_abeta42 = st.number_input("C2N Plasma Aβ42", value=None, min_value=0.0, 
                                                key="c2n_plasma_abeta42")
            c2n_plasma_abeta42_abeta40 = st.number_input("C2N Plasma Aβ42/Aβ40 Ratio",value=None,  min_value=0.0,
                                                        key="c2n_plasma_abeta42_abeta40")
            c2n_plasma_ptau217 = st.number_input("C2N Plasma pTau217",value=None, min_value=0.0,
                                                key="c2n_plasma_ptau217")
            roche_plasma_gfap = st.number_input("Roche Plasma GFAP",value=None, min_value=0.0,
                                               key="roche_plasma_gfap")
            c2n_plasma_nptau217 = st.number_input("C2N Plasma npTau217", value=None, min_value=0.0,
                                                 key="c2n_plasma_nptau217")
            roche_plasma_nfl = st.number_input("Roche Plasma NfL", value=None,  min_value=0.0,
                                             key="roche_plasma_nfl")

        submit_button = st.form_submit_button("Submit Data")
        st.markdown('</div>', unsafe_allow_html=True)

        if submit_button:
            if not ptid or not name:
                st.error("Patient ID and Name are required fields!")
            else:
                # Prepare data for database
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
                    'CDRSB' : cdrsb,
                    'FAQ': faq,
                    'APOE4': apoe4,
                    'PTETHCAT': ptethcat,
                    'PTEDUCAT' : ptedu,
                    'PTGENDER': ptgender
                }
                
                if add_patient(data):
                    st.success("Patient data successfully added!")
                else:
                    st.error("Error: Patient ID already exists!")

with tab2:
    st.header("Patient Records")
    
    # Add search/filter options
    search_col1, search_col2, delete_col = st.columns([2, 1, 1])
    with search_col1:
        search_term = st.text_input("Search by Patient ID or Name")
    with search_col2:
        sort_by = st.selectbox("Sort by", ["PTID", "NAME", "AGE", "timestamp"])
    
    # Get and display data
    df = get_all_patients()
    
    # Apply search filter
    if search_term:
        df = df[df['PTID'].str.contains(search_term, case=False) | 
                df['NAME'].str.contains(search_term, case=False)]
    
    # Sort data
    df = df.sort_values(by=sort_by)
    
    # Prepare data for display with selection
    df_display = df.copy()
    
    # Add selection column
    df_display['Select'] = False
    
    # Create an editable dataframe with selection
    edited_df = st.data_editor(
        df_display, 
        column_config={
            "Select": st.column_config.CheckboxColumn(
                "Delete",
                default=False
            )
        },
        disabled=df_display.columns.drop('Select').tolist(),
        use_container_width=True
    )
    
    # Delete selected records
    if st.button("🗑️ Delete Selected Records"):
        selected_ptids = edited_df[edited_df['Select']]['ID'].tolist()
        
        if not selected_ptids:
            st.warning("No records selected!")
        else:
            with st.status("Processing deletions...", expanded=True) as status:
                deleted_count = 0
                for ptid in selected_ptids:
                    if delete_patient(ptid):
                        deleted_count += 1
                status.update(label=f"Deleted {deleted_count} records", state="complete")
            st.rerun()
    
    # Export options
    if not df.empty:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"patient_data_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
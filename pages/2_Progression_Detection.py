import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from utils import apply_custom_css
from database import get_all_patients, get_patient

def get_patients_with_multiple_visits():
    df = get_all_patients()
    visit_counts = df.groupby('PTID').size()
    eligible_ptids = visit_counts[visit_counts >= 2].index.tolist()
    return eligible_ptids

def create_biomarker_timeline(df):
    # Create a figure with multiple biomarker trends
    fig = go.Figure()
    
    biomarkers = {
        'C2N_plasma_ptau217': 'pTau217',
        'C2N_plasma_Abeta42': 'Aβ42',
        'Roche_plasma_GFAP': 'GFAP',
        'Roche_plasma_NfL': 'NfL'
    }
    
    for column, name in biomarkers.items():
        if column in df.columns:
            fig.add_trace(go.Scatter(
                x=df['VISCODE'],
                y=df[column],
                name=name,
                mode='lines+markers'
            ))
    
    fig.update_layout(
        title='Biomarker Progression Over Time',
        xaxis_title='Visit',
        yaxis_title='Biomarker Levels',
        height=400,
        template='plotly_white'
    )
    return fig

def create_cognitive_scores_radar(df):
    # Latest visit scores
    latest_visit = df.iloc[-1]
    
    # Create radar chart for cognitive scores
    categories = ['MMSE', 'ADAS11', 'ADAS13', 'FAQ']
    values = [latest_visit['MMSE'], latest_visit['ADAS11'], 
              latest_visit['ADAS13'], latest_visit['FAQ']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Latest Scores'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.2]
            )),
        showlegend=False,
        title='Current Cognitive Status',
        height=400
    )
    return fig

def create_cognitive_trend(df):
    # Create line chart for cognitive scores over time
    fig = go.Figure()
    
    cognitive_measures = {
        'MMSE': 'Mini-Mental State Examination',
        'ADAS11': 'ADAS-Cog 11',
        'ADAS13': 'ADAS-Cog 13',
        'FAQ': 'Functional Activities Questionnaire'
    }
    
    for score, label in cognitive_measures.items():
        if score in df.columns:
            fig.add_trace(go.Scatter(
                x=df['VISCODE'],
                y=df[score],
                name=label,
                mode='lines+markers'
            ))
    
    fig.update_layout(
        title='Cognitive Scores Progression',
        xaxis_title='Visit',
        yaxis_title='Score',
        height=400,
        template='plotly_white'
    )
    return fig

def create_biomarker_correlation_heatmap(df):
    # Create correlation matrix for biomarkers
    biomarkers = ['C2N_plasma_ptau217', 'C2N_plasma_Abeta42', 
                  'Roche_plasma_GFAP', 'Roche_plasma_NfL']
    corr_matrix = df[biomarkers].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=biomarkers,
        y=biomarkers,
        colorscale='RdBu',
        zmin=-1, zmax=1
    ))
    
    fig.update_layout(
        title='Biomarker Correlations',
        height=400
    )
    return fig

def create_visit_intervals_analysis(df):
    # Calculate time between visits
    df['visit_date'] = pd.to_datetime(df['timestamp'])
    df['days_between_visits'] = df['visit_date'].diff().dt.days
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['VISCODE'][1:],
        y=df['days_between_visits'][1:],
        name='Days between visits'
    ))
    
    fig.update_layout(
        title='Visit Intervals Analysis',
        xaxis_title='Visit',
        yaxis_title='Days since previous visit',
        height=400
    )
    return fig

def create_biomarker_box_plot(df):
    # Prepare data for box plot
    biomarkers = ['C2N_plasma_ptau217', 'C2N_plasma_Abeta42', 
                  'Roche_plasma_GFAP', 'Roche_plasma_NfL']
    
    fig = go.Figure()
    for biomarker in biomarkers:
        fig.add_trace(go.Box(
            y=df[biomarker],
            name=biomarker,
            boxpoints='all'
        ))
    
    fig.update_layout(
        title='Biomarker Distribution Analysis',
        yaxis_title='Value',
        height=400
    )
    return fig

def create_cognitive_decline_rate(df):
    # Calculate rate of change for cognitive scores
    df['MMSE_change'] = df['MMSE'].diff() / df['days_between_visits']
    df['ADAS11_change'] = df['ADAS11'].diff() / df['days_between_visits']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['VISCODE'][1:],
        y=df['MMSE_change'][1:],
        name='MMSE Change Rate',
        mode='lines+markers'
    ))
    fig.add_trace(go.Scatter(
        x=df['VISCODE'][1:],
        y=df['ADAS11_change'][1:],
        name='ADAS11 Change Rate',
        mode='lines+markers'
    ))
    
    fig.update_layout(
        title='Rate of Cognitive Decline',
        xaxis_title='Visit',
        yaxis_title='Change per day',
        height=400
    )
    return fig

def create_combined_score_analysis(df):
    # Create a combined score using multiple metrics
    df['combined_score'] = (
        (df['MMSE'] / df['MMSE'].max()) -
        (df['ADAS11'] / df['ADAS11'].max()) -
        (df['FAQ'] / df['FAQ'].max())
    )
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['VISCODE'],
        y=df['combined_score'],
        mode='lines+markers',
        name='Combined Score'
    ))
    
    fig.update_layout(
        title='Combined Performance Score Trend',
        xaxis_title='Visit',
        yaxis_title='Combined Score (normalized)',
        height=400
    )
    return fig


def display_prediction_results(cn_prob, mci_prob, ad_prob):
    # Create columns for each prediction
    col1, col2, col3 = st.columns(3)
    
    # Helper function to determine color based on probability
    def get_color(prob):
        if prob < 30:
            return "green"
        elif prob < 70:
            return "orange"
        else:
            return "red"
    
    # Display each probability with custom styling
    with col1:
        st.markdown(
            f"""
            <div style='background-color: rgba(0, 128, 0, 0.1); padding: 20px; border-radius: 10px; text-align: center;'>
                <h3 style='margin: 0; color: {get_color(cn_prob)};'>CN</h3>
                <h2 style='margin: 0; color: {get_color(cn_prob)};'>{cn_prob}%</h2>
                <p style='margin: 0; font-size: 0.8em;'>Cognitively Normal</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div style='background-color: rgba(255, 165, 0, 0.1); padding: 20px; border-radius: 10px; text-align: center;'>
                <h3 style='margin: 0; color: {get_color(mci_prob)};'>MCI</h3>
                <h2 style='margin: 0; color: {get_color(mci_prob)};'>{mci_prob}%</h2>
                <p style='margin: 0; font-size: 0.8em;'>Mild Cognitive Impairment</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div style='background-color: rgba(255, 0, 0, 0.1); padding: 20px; border-radius: 10px; text-align: center;'>
                <h3 style='margin: 0; color: {get_color(ad_prob)};'>AD</h3>
                <h2 style='margin: 0; color: {get_color(ad_prob)};'>{ad_prob}%</h2>
                <p style='margin: 0; font-size: 0.8em;'>Alzheimer's Disease</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Apply custom CSS
apply_custom_css()

st.title("📈 Progression Detection")
st.markdown("""
    Analyze disease progression for patients with multiple visits.
    Select a patient ID from the dropdown menu and click 'Analyze' to view the progression analysis.
""")

# Create a container for the selection interface
selection_container = st.container()

with selection_container:
    col1, col2 = st.columns([3, 1])
    
    # Get list of eligible patients
    eligible_patients = get_patients_with_multiple_visits()
    
    with col1:
        selected_patient = st.selectbox(
            "Select Patient ID",
            options=eligible_patients,
            help="Only showing patients with 2 or more visits"
        )
    
    with col2:
        analyze_button = st.button("Analyze", type="primary")

# Handle analysis when button is clicked
if analyze_button and selected_patient:
    try:
        # Display model prediction results first
        st.markdown("### 🤖 Model Prediction Results")
        st.markdown("Based on the patient's progression pattern, the percentage of next visit diagnosis is:")
        
        # Using dummy values - replace these with actual model predictions
        display_prediction_results(
            cn_prob=6,
            mci_prob=60,
            ad_prob=34
        )
        
        # Add a separator
        st.markdown("---")
        
        # Continue with the rest of the analysis
        st.info("Running detailed analysis...")
        patient_data = get_patient(selected_patient)
        
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Biomarkers", "Cognitive Scores", "Advanced Analysis", 
            "Rate of Change", "Correlations"
        ])
        
        with tab1:
            st.plotly_chart(create_biomarker_timeline(patient_data), use_container_width=True)
            st.plotly_chart(create_biomarker_box_plot(patient_data), use_container_width=True)
            
        with tab2:
            st.plotly_chart(create_cognitive_trend(patient_data), use_container_width=True)
            st.plotly_chart(create_cognitive_scores_radar(patient_data), use_container_width=True)
            
        with tab3:
            st.plotly_chart(create_combined_score_analysis(patient_data), use_container_width=True)
            st.plotly_chart(create_visit_intervals_analysis(patient_data), use_container_width=True)
            
        with tab4:
            st.plotly_chart(create_cognitive_decline_rate(patient_data), use_container_width=True)
            
        with tab5:
            st.plotly_chart(create_biomarker_correlation_heatmap(patient_data), use_container_width=True)
            
        # Add statistical summary
        st.subheader("Statistical Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Biomarker Variability")
            biomarker_stats = patient_data[[
                'C2N_plasma_ptau217', 'C2N_plasma_Abeta42', 
                'Roche_plasma_GFAP', 'Roche_plasma_NfL'
            ]].describe()
            st.dataframe(biomarker_stats)
            
        with col2:
            st.write("Cognitive Score Variability")
            cognitive_stats = patient_data[['MMSE', 'ADAS11', 'ADAS13', 'FAQ']].describe()
            st.dataframe(cognitive_stats)
            
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
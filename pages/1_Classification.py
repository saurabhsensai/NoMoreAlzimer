import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib 
from utils import apply_custom_css, make_classification_prediction
# Apply custom CSS
apply_custom_css()

st.title("🔍 Classification Analysis")
st.markdown("""
    This page analyzes various biomarkers and clinical data to provide classification predictions.
    Please input the required parameters below.
""")

# Create two columns for better layout
col1, col2 = st.columns(2)

# Input form
with st.form("classification_form"):
    st.subheader("📋 Input Parameters")
    
    with col1:
        st.markdown("### Plasma Biomarkers")
        c2n_plasma_ptau217_ratio = st.number_input("C2N Plasma pTau217 Ratio", min_value=0.0)
        c2n_plasma_abeta42 = st.number_input("C2N Plasma Aβ42", min_value=0.0)
        c2n_plasma_abeta42_abeta40 = st.number_input("C2N Plasma Aβ42/Aβ40 Ratio", min_value=0.0)
        c2n_plasma_ptau217 = st.number_input("C2N Plasma pTau217", min_value=0.0)
        roche_plasma_gfap = st.number_input("Roche Plasma GFAP", min_value=0.0)
        c2n_plasma_nptau217 = st.number_input("C2N Plasma npTau217", min_value=0.0)
        roche_plasma_nfl = st.number_input("Roche Plasma NfL", min_value=0.0)
        
    with col2:
        st.markdown("### Clinical Parameters")
        age = st.number_input("Age", min_value=0, max_value=120)
        mmse = st.number_input("MMSE Score", min_value=0, max_value=30)
        adas11 = st.number_input("ADAS11 Score", min_value=0.0)
        adas13 = st.number_input("ADAS13 Score", min_value=0.0)
        faq = st.number_input("FAQ Score", min_value=0.0)
        apoe4 = st.selectbox("APOE4 Status", options=[0, 1, 2])
        ptethcat = st.selectbox("Ethnicity Category", options=['Not Hispanic/Latino', 'Hispanic/Latino'])
        ptgender = st.selectbox("Gender", options=['Male', 'Female'])

    submit_button = st.form_submit_button("Generate Classification")

# Handle prediction when form is submitted
if submit_button:
    input_data = {
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
        'PTGENDER': ptgender
    }
    
    prediction = make_classification_prediction(input_data)
    
    if prediction is not None:
        st.markdown("### 🎯 Classification Results")
        
        # Define classes and colors
        class_names = ['CN', 'MCI', 'AD']
        colors = ['#2ecc71', '#f1c40f', '#e74c3c']
        
        # 1. Main Prediction Display
        max_prob_idx = np.argmax(prediction)
        max_prob_class = class_names[max_prob_idx]
        
        # Create three columns for the main prediction display
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
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title="Probability Distribution (Radar View)"
        )
        
        # 3. Gauge Chart for Confidence
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
        
        # 5. Create tabs for different visualizations
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
            
            styled_df = results_df.style.format({
                'Probability': '{:.2%}'
            }).background_gradient(subset=['Probability'], cmap='Blues')
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Feature importance visualization (if available)
            st.markdown("### Input Features Summary")
            features_df = pd.DataFrame([input_data])
            styled_features = features_df.style.background_gradient(cmap='viridis')
            st.dataframe(styled_features, use_container_width=True)
        
        # 6. Additional Information
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
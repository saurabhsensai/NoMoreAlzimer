import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from progression_pytorch import LongitudinalModel, preprocess_inference_data, process_data
import joblib

def apply_custom_css():
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            margin-top: 2rem;
        }
        .prediction-box {
            padding: 2rem;
            border-radius: 0.5rem;
            background-color: #f8f9fa;
            margin-top: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

def preprocess_input_data(input_data):
    """
    Preprocess input data to match model requirements
    """
    # Convert input dictionary to DataFrame
    df = pd.DataFrame([input_data])
    
    # Categorical encoding
    df['PTETHCAT'] = pd.Categorical(
        df['PTETHCAT'], 
        categories=['Not Hispanic/Latino', 'Hispanic/Latino', 'Unknown']
    ).codes
    
    df['PTGENDER'] = pd.Categorical(
        df['PTGENDER'], 
        categories=['Female', 'Male']
    ).codes
    
    # Ensure correct column order
    columns_order = [
        'C2N_plasma_ptau217_ratio',
        'C2N_plasma_Abeta42',
        'C2N_plasma_Abeta42_Abeta40',
        'C2N_plasma_ptau217',
        'Roche_plasma_GFAP',
        'C2N_plasma_nptau217',
        'Roche_plasma_NfL',
        'AGE',
        'MMSE',
        'ADAS11',
        'ADAS13',
        'FAQ',
        'APOE4',
        'PTETHCAT',
        'PTGENDER'
    ]
    df = df[columns_order]
    
    # Load the pre-fitted scaler
    try:
        scaler = joblib.load('Models/scaler.pkl')
    except FileNotFoundError:
        # If scaler is not found, create a new one (for development/testing)
        scaler = StandardScaler()
        
    # Scale the features
    scaled_data = scaler.transform(df)
    
    # Reshape to (1, 15) as required by the model
    scaled_data = scaled_data.reshape(1, -1)
    
    return scaled_data

def make_classification_prediction(input_data):
    """
    Preprocess data and make classification prediction
    """
    try:
        # Preprocess the input data
        
        processed_data = preprocess_input_data(input_data)
        print(processed_data.shape)
        
        # Load the model
        model = load_model('Models/3class_classification_model.h5')
        
        # Make prediction
        prediction = model.predict(processed_data)
        # pred = np.argmax(prediction, axis=1)
        
        return prediction[0]  # Return probabilities for each class
        
    except Exception as e:
        st.error(f"Error in making prediction: {str(e)}")
        return None

def make_progression_prediction(input_data):
    # Implementation for progression prediction
    pass


def add_extra_column(df):
    df.loc[:, 'tr_label'] = pd.Categorical(df['DX'], categories=['CN', 'MCI', 'Dementia']).codes
    df['PTGENDER'] = pd.Categorical(df['PTGENDER'], categories=['Female' ,'Male']).codes
    
    df['PTETHCAT'] = pd.Categorical(df['PTETHCAT'], categories=['Not Hispanic/Latino', 'Hispanic/Latino', 'Unknown']).codes
  
    return df


def get_progression_results(df):

    try:
        scaler = joblib.load('progression_model/pr-scaler.pkl')
    except FileNotFoundError:
        # If scaler is not found, create a new one (for development/testing)
        print("Error while loading scalar")

    means = scaler.mean_
    stds = scaler.scale_

    le = LabelEncoder()

    time_steps = 2 
    X_test_lon_ex , X_test_dem_ex, y_test_ex = process_data(df, means, stds, scaler,  time_steps)
    y_test_ex = le.fit_transform(y_test_ex)

    return 24

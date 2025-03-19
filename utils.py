import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from progression_pytorch import LongitudinalModel, preprocess_inference_data, process_data
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import pandas as pd

device = 'cpu'





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
        'PTGENDER',
        'PTEDUCAT'
    ]


    df = df[columns_order]
    df.fillna(np.nan, inplace=True)
    
    np_array = df.values
    
    
    
    

    
    return np_array





def make_classification_prediction(input_data):
    """
    Preprocess data and make classification prediction
    """
    try:
        # Preprocess the input data
        
        np_array = preprocess_input_data(input_data)
        print(np_array)
        print("Working till here................")
        model2 = joblib.load('XgBoost_Model/xgboost_model_classification.joblib')
        print("Working till here................")
        y_pred = model2.predict_proba(np_array)

        return y_pred[0]
        
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


def do_preprocessing(X_test_lon_ex , X_test_dem_ex):
    X_test_lon_ex = torch.FloatTensor(X_test_lon_ex)
    X_test_dem_ex = torch.FloatTensor(X_test_dem_ex)

    return X_test_lon_ex , X_test_dem_ex


def predict_single(loaded_model, lon_data, dem_data):
    # Convert to tensor
    lon_tensor = torch.FloatTensor(lon_data).unsqueeze(0)  # Add batch dim
    dem_tensor = torch.FloatTensor(dem_data).unsqueeze(0)
    
    # Move to device
    lon_tensor = lon_tensor.to(device)
    dem_tensor = dem_tensor.to(device)
    
    # Inference
    loaded_model.eval()
    with torch.no_grad():
        output = loaded_model(lon_tensor, dem_tensor)
        prob = F.softmax(output, dim=1)
    
    return prob.cpu().numpy().squeeze()

def get_progression_results(df):

    try:
        scaler = joblib.load('progression_model/pr-scaler-6months.pkl')
    except FileNotFoundError:
        # If scaler is not found, create a new one (for development/testing)
        print("Error while loading scalar")

    means = scaler.mean_
    stds = scaler.scale_


    time_steps = 2 
    X_test_lon_ex , X_test_dem_ex = process_data(df, means, stds, scaler,  time_steps)


    X_test_lon_ex , X_test_dem_ex = do_preprocessing(X_test_lon_ex , X_test_dem_ex)

    loaded_model = LongitudinalModel(
    num_features=14, 
    dem_features=4,
    num_classes=3,
    time_steps=2
    )
    
    loaded_model.load_state_dict(torch.load("progression_model/pytorch_progression_alldata_2_6_months.pth", map_location=device))
    loaded_model.to(device)

    sample_lon = X_test_lon_ex[0].cpu() 
    sample_dem = X_test_dem_ex[0].cpu()
    probabilities = predict_single(loaded_model, sample_lon, sample_dem)

    return probabilities

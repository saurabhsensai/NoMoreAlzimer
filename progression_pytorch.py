import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


plasma_features = ["C2N_plasma_ptau217_ratio", 'C2N_plasma_Abeta42', 'C2N_plasma_Abeta42_Abeta40',
                   'C2N_plasma_ptau217', 'Roche_plasma_GFAP', "C2N_plasma_nptau217",
                   'Roche_plasma_NfL', 'AGE', 'MMSE', 'ADAS11', 'ADAS13', 'CDRSB', 'FAQ' , 'tr_label']
demographic_features = ['APOE4', 'PTEDUCAT', 'PTETHCAT', 'PTGENDER']
label_column = 'DX'  # Replace with your label column name


class LongitudinalModel(nn.Module):
    def __init__(self, num_features, dem_features, num_classes, time_steps):
        super().__init__()
        self.num_features = num_features
        self.dem_features = dem_features
        self.num_classes = num_classes
        self.time_steps = time_steps

        # Longitudinal processing
        self.gru1 = nn.GRU(
            input_size=num_features,
            hidden_size=16,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(0.2)
        self.gru2 = nn.GRU(
            input_size=16,
            hidden_size=8,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(0.2)

        # Demographic processing
        self.classifier = nn.Sequential(
            nn.Linear(8 + dem_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
            nn.LogSoftmax(dim=1)  # Added LogSoftmax for stability
        )

    def forward(self, input_lon, input_dem):
        # Mask invalid timesteps
        input_lon = input_lon.masked_fill(input_lon == -1.0, 0)

        # Compute valid timesteps
        mask = (input_lon != 0).any(dim=-1)
        lengths = mask.sum(dim=1)

        # Handle empty sequences
        if (lengths == 0).any():
            raise ValueError("All features masked in some samples")

        # Sort sequences by length
        lengths_sorted, sort_idx = torch.sort(lengths, descending=True)
        input_lon = input_lon[sort_idx]
        input_dem = input_dem[sort_idx]

        # Pack and process through GRUs
        packed = pack_padded_sequence(input_lon, lengths_sorted.cpu(), 
                                      batch_first=True, enforce_sorted=True)

        # First GRU layer
        packed_out, _ = self.gru1(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.dropout1(out)
        out = F.relu(out)

        # Second GRU layer
        packed_out = pack_padded_sequence(out, lengths_sorted.cpu(),
                                           batch_first=True, enforce_sorted=True)
        _, hidden = self.gru2(packed_out)
        x = hidden[-1]  # Get last hidden state
        x = self.dropout2(x)

        # Concatenate with demographic data
        combined = torch.cat([x, input_dem], dim=1)

        # Classification layers
        logits = self.classifier(combined)

        # Restore original order
        _, unsort_idx = torch.sort(sort_idx)
        return logits[unsort_idx]

    def get_regularization_loss(self):
        """L2 regularization for GRU input weights (matching Keras implementation)"""
        reg_loss = 0.0
        l2_lambda = 0.001
        for name, param in self.named_parameters():
            if 'weight' in name:
                reg_loss += l2_lambda * torch.sum(param ** 2)
        return reg_loss


def preprocess_inference_data(X_missing,scaler, means, stds):
    X_imputed = X_missing.copy()
    
    X_imputed_values = X_imputed.values  
    missing_mask = np.isnan(X_imputed_values) 
    

    for i in range(X_imputed_values.shape[1]):
        X_imputed_values[missing_mask[:, i], i] = means[i] - stds[i]
    X_imputed = pd.DataFrame(X_imputed_values, columns=X_missing.columns)
    
    X_scaled = scaler.transform(X_imputed)
    return X_scaled

def process_data(df, means, stds, scaler, time_steps=2):
    longitudinal_data = []
    demographic_data = []
    
    
    grouped = df.groupby('PTID')
    for rid, group in grouped:
        scaled_plasma = preprocess_inference_data(group[plasma_features],scaler, means, stds)
    
        if len(group) < time_steps:
            padded = np.zeros((time_steps, len(plasma_features)))
            padded[:len(group), :] = scaled_plasma
            longitudinal_data.append(padded)
        else:
            longitudinal_data.append(scaled_plasma[:time_steps, :])
    
        demographic_data.append(group[demographic_features].iloc[0].values)
        
    
    return np.array(longitudinal_data), np.array(demographic_data)
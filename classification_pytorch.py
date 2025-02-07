import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

class BiLSTMModel(nn.Module):
    def __init__(self, dem_features, num_classes):
        super(BiLSTMModel, self).__init__()
        # Longitudinal processing layers
        self.bilstm1 = nn.LSTM(
            input_size=12,
            hidden_size=16,
            bidirectional=True,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(0.2)
        self.bilstm2 = nn.LSTM(
            input_size=32,  # 16*2 for bidirectional
            hidden_size=8,
            bidirectional=True,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(0.2)
        
        # Demographic and classification layers
        self.dense1 = nn.Linear(16 + dem_features, 32)  # 8*2 from BiLSTM + dem features
        self.dense2 = nn.Linear(32, 16)
        self.dense3 = nn.Linear(16, num_classes)
        
        # Weight initialization
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
                
    def forward(self, input_lon, input_dem):
        # Calculate valid lengths for masking
        mask = (input_lon == -1.0).all(dim=-1)
        valid_lengths = (~mask).sum(dim=1)
        
        # Check for completely masked samples
        if (valid_lengths == 0).any():
            raise ValueError("Some samples have all longitudinal features masked.")
        
        # Sort inputs by valid lengths
        sorted_lengths, sort_idx = valid_lengths.sort(descending=True)
        input_lon_sorted = input_lon[sort_idx]
        input_dem_sorted = input_dem[sort_idx]
        
        # Pack padded sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(
            input_lon_sorted, sorted_lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        
        # First BiLSTM layer
        packed_output, _ = self.bilstm1(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = self.dropout1(output)
        
        # Second BiLSTM layer
        packed_output = nn.utils.rnn.pack_padded_sequence(
            output, sorted_lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        _, (hidden, _) = self.bilstm2(packed_output)
        
        # Concatenate final hidden states
        bilstm_out = torch.cat((hidden[-2], hidden[-1]), dim=1)
        bilstm_out = self.dropout2(bilstm_out)
        
        # Combine with demographic data
        combined = torch.cat((bilstm_out, input_dem_sorted), dim=1)
        
        # Classification layers
        x = F.relu(self.dense1(combined))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        
        # Restore original batch order
        _, unsort_idx = sort_idx.sort()
        return x[unsort_idx]
    


plasma_features_ML = ["C2N_plasma_ptau217_ratio", 'C2N_plasma_Abeta42', 'C2N_plasma_Abeta42_Abeta40',
                   'C2N_plasma_ptau217', 'Roche_plasma_GFAP', "C2N_plasma_nptau217",
                   'Roche_plasma_NfL', 'AGE', 'MMSE', 'ADAS11', 'ADAS13', 'FAQ']


demographic_features_ML = ['APOE4', 'PTEDUCAT', 'PTETHCAT', 'PTGENDER']


def preprocess_inference_data_ML(X_missing,scaler, means, stds):
    X_imputed = X_missing.copy()
    
    X_imputed_values = X_imputed.values  
    missing_mask = np.isnan(X_imputed_values) 
    

    for i in range(X_imputed_values.shape[1]):
        X_imputed_values[missing_mask[:, i], i] = means[i] - stds[i]
    X_imputed = pd.DataFrame(X_imputed_values, columns=X_missing.columns)
    
    X_scaled = scaler.transform(X_imputed)
    return X_scaled





def process_data_ML(df, scaler, means, stds, time_steps=1):
    longitudinal_data = []
    demographic_data = []
    
    
    grouped = df.groupby('RID')
    for rid, group in grouped:
        scaled_plasma = preprocess_inference_data_ML(group[plasma_features_ML],scaler, means, stds)
    
        if len(group) < time_steps:
            padded = np.zeros((time_steps, len(plasma_features_ML)))
            padded[:len(group), :] = scaled_plasma
            longitudinal_data.append(padded)
        else:
            longitudinal_data.append(scaled_plasma[:time_steps, :])
    
        demographic_data.append(group[demographic_features_ML].iloc[0].values)
        
    
    return np.array(longitudinal_data), np.array(demographic_data)

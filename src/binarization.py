import pandas as pd

def create_binary_features(data):
    binary_features = pd.DataFrame()
    
    # Age bins
    binary_features['age_30plus'] = (data['age'] >= 30).astype(int)
    binary_features['age_45plus'] = (data['age'] >= 45).astype(int)
    binary_features['age_60plus'] = (data['age'] >= 60).astype(int)
    
    # Blood pressure bins
    binary_features['bp_120plus'] = (data['trestbps'] >= 120).astype(int)
    binary_features['bp_140plus'] = (data['trestbps'] >= 140).astype(int)
    binary_features['bp_160plus'] = (data['trestbps'] >= 160).astype(int)
    
    # Chest pain type (one-hot)
    cp_dummies = pd.get_dummies(data['cp'], prefix='cp')
    binary_features = pd.concat([binary_features, cp_dummies], axis=1)
    
    # Ensure all values are proper binary values
    binary_features = binary_features.astype(int)
    
    return binary_features

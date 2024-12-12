import pandas as pd
from sklearn.model_selection import train_test_split

def load_heart_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    data = pd.read_csv(url, names=columns, na_values='?')
    data = data.dropna()
    data['target'] = (data['target'] > 0).astype(int)
    return data

def get_train_test_split(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)

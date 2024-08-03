# tests/test_feature_engineering.py
import pandas as pd
from src.feature_engineering import create_features

def test_create_features():
    data = {
        'cpu_usage': [0.5, 0.6],
        'memory_usage': [0.7, 0.8]
    }
    df = pd.DataFrame(data)
    feature_df = create_features(df)

    assert 'cpu_memory_ratio' in feature_df.columns
    assert feature_df['cpu_memory_ratio'][0] == 0.5 / 0.7

# tests/test_data_preprocessing.py
import pandas as pd
from src.data_preprocessing import preprocess_data

def test_preprocess_data():
    data = {
        'cpu_usage': [50, 55, None, 60],
        'memory_usage': [70, 75, 80, None]
    }
    df = pd.DataFrame(data)
    processed_df = preprocess_data(df)

    assert processed_df.isnull().sum().sum() == 0
    assert processed_df['cpu_usage'].mean() == 0
    assert processed_df['memory_usage'].mean() == 0

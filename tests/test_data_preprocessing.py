import unittest
from src.data_preprocessing import preprocess_data
import pandas as pd

class TestDataPreprocessing(unittest.TestCase):
    def test_preprocess_data(self):
        data = {'cpu_usage': [1.0], 'memory_usage': [2.0], 'cpu_memory_ratio': [0.5], 'target_metric': [4]}
        df = pd.DataFrame(data)
        X, y = preprocess_data(df)
        self.assertEqual(X.shape, (1, 3))
        self.assertEqual(len(y), 1)

if __name__ == '__main__':
    unittest.main()

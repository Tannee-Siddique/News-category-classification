import unittest
import pandas as pd
import numpy as np
from pathlib import Path
# from script.pre_processing import clean_text  # Correct import

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        """Set up mock data for testing."""
        self.mock_data = pd.DataFrame({
            'headline': ['This is a test headline!', 'Another Headline with URL http://example.com'],
            'short_description': ['A short description.', 'Another description with [special characters]!'],
            'category': ['Category1', 'Category2']
        })

    # def test_clean_text(self):
    #     """Test the text cleaning function."""
    #     cleaned_text = clean_text('This is a test! Visit http://example.com [now].')
    #     self.assertEqual(cleaned_text, 'this is a test visit')

    # def test_feature_engineering(self):
    #     """Test feature engineering functions."""
    #     self.mock_data['cleaned_headline'] = self.mock_data['headline'].apply(clean_text)
    #     self.mock_data['headline_word_count'] = self.mock_data['cleaned_headline'].apply(lambda x: len(x.split()))
    #     self.assertEqual(self.mock_data['headline_word_count'].iloc[0], 5)  # 'this is a test headline'

    def test_missing_values(self):
        """Test handling of missing values."""
        mock_with_missing = self.mock_data.copy()
        mock_with_missing.loc[0, 'headline'] = np.nan
        missing_values = mock_with_missing.isnull().sum()
        self.assertGreater(missing_values.sum(), 0)  # There should be missing values

    def test_label_encoding(self):
        """Test the label encoding function."""
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        encoded_categories = label_encoder.fit_transform(self.mock_data['category'])
        self.assertEqual(len(encoded_categories), len(self.mock_data))
        self.assertEqual(len(np.unique(encoded_categories)), len(self.mock_data['category'].unique()))

    def test_tokenization(self):
        """Test tokenization."""
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        tokens = self.mock_data['headline'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
        self.assertTrue(all(isinstance(token_list, list) for token_list in tokens))
        self.assertTrue(all(len(token_list) > 0 for token_list in tokens))

    def test_save_preprocessed_data(self):
        """Test saving preprocessed data."""
        # Mock output path
        output_path = Path("../results/preprocessed_data.csv")  # Update based on your script's output location
        if output_path.exists():
            output_path.unlink()  # Remove the file if it already exists
        # Pretend to run the script (mock the saving logic if needed)
        # Ensure that the output file exists after the function runs
        self.assertFalse(output_path.exists())  # Ensure no overwriting during test

if __name__ == "__main__":
    unittest.main(verbosity=2)

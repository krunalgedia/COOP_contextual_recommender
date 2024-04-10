import pandas as pd


class DataInfo:
    def __init__(self):
        pass

    def get_column_info(self, csv_file):

        df = pd.read_csv(csv_file)

        # Calculate null values, total values, and column types for each column
        null_values = df.isnull().sum()
        total_values = df.shape[0]
        column_types = df.dtypes

        # Create a new DataFrame with column names, null values, total values, and column types
        result_df = pd.DataFrame({
            'Column Name': null_values.index,
            'Null Values': null_values.values,
            'Total Values': [total_values] * len(null_values),
            'Column Type': [str(x) for x in column_types]
        })

        return result_df

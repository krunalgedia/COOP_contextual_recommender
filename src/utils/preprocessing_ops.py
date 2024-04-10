import pandas as pd

class PreprocessUtils:
    def __init__(self, df):
        self.df = df

    @classmethod
    def preprocess_dataframe(cls, df_path, model, name):
        df = pd.read_csv(df_path)

        # Drop rows where 'title' is missing
        df = df.dropna(subset=['title'])

        # Add 'type' column
        df.loc[:, 'type'] = name

        # Extract 'rate' and 'votes' columns from 'rating' column
        df.loc[:, 'rate'] = df['rating'].str.extract(r'Rated  of (\d+)')
        df.loc[:, 'votes'] = df['rating'].str.extract(r'\((\d+)\)')

        # Format 'price' column
        if df.price.dtype == 'object':
            df['price'] = df['price'].str.extract(r'(\d+\.?\d*)').astype(float)
        df.loc[:, 'price_str'] = df['price'].apply(lambda x: f'Price is {x:.2f}')

        # Combine 'title' and 'price' columns into 'content' column
        df.loc[:, 'content'] = df['title'].astype(str) + '.' + df['price_str']

        # Encode 'title' column and store in 'embedding' column
        df.loc[:, 'embedding'] = df['title'].apply(lambda x: model.encode(str(x)))

        return cls(df)

    @classmethod
    def concat_dataframe(cls, cls_list):
        df = pd.concat([i.df for i in cls_list], axis=0)
        df.reset_index(drop=True, inplace=True)
        return cls(df)

    @classmethod
    def split_embeddings(cls, dfcls):
        split_df = dfcls.df['embedding'].apply(pd.Series).add_prefix('embedding_')
        df = pd.concat([dfcls.df, split_df], axis=1)
        return cls(df)
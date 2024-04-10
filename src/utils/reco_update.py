import pandas as pd
from umap import UMAP
from sklearn.cluster import KMeans
import faiss
import numpy as np
import os
import sys
from joblib import dump, load
# Add the path to the other directory to the Python path
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../models'))
from src.configs.config import ModelConfig, PathsConfig
from src.utils.preprocessing_ops import PreprocessUtils
from src.utils.file_ops import FilesOps
import joblib
from faiss import write_index, read_index
import pickle

class RecoUpdate():
    def __init__(self, isCurrentIsRoot=False):
        self.path = PathsConfig(isCurrentIsRoot)
        self.model = ModelConfig()
        self.centroid_index = read_index(str(self.path.centroid_index))
        self.cluster_index_dict = pickle.load(open(self.path.cluster_index_dict, 'rb'))
        self.dim_reduction_pipeline = joblib.load(self.path.final_dim_reduction_pipeline)
        self.kmeans_pipeline = joblib.load(self.path.kmeans_joblib)
        self.model = self.model.model
        self.emb_split_df = pd.read_pickle(self.path.df_pickle)

    def recommend_catalog(self, query_string):

        # Encode the query string using the model
        query = self.model.encode(query_string, convert_to_tensor=True)

        # Convert query point to numpy array and float32 (required by FAISS)
        query_point = query.numpy().astype(np.float32)

        # Transform query point using PCA pipeline
        query_point_pca = self.dim_reduction_pipeline.named_steps['clf'].transform(query_point.reshape(1, -1))

        # Perform nearest neighbor search
        _, nearest_centroid_index = self.centroid_index.search(query_point_pca, k=1)
        _, nearest_cluster_index = self.cluster_index_dict[nearest_centroid_index[0][0]].search(query_point_pca, k=3)

        # Get the indices of recommended rows
        recommended_indices = nearest_cluster_index[0].tolist()

        # Retrieve recommended rows from the DataFrame
        recommended_rows = self.emb_split_df.loc[self.emb_split_df['klabels'] == nearest_centroid_index[0][0]].iloc[
            recommended_indices]

        return recommended_rows

    def update_catalog(self, title, typ, quantity, price, isCurrentIsRoot=True):

        data = {
            'title': title,
            'type': typ,
            'quantity': quantity,
            'price': price
        }

        # Create the DataFrame
        df = pd.DataFrame(data, index=[0])
        df.loc[:, 'embedding'] = df['title'].apply(lambda x: self.model.encode(str(x)))
        emb_df = df['embedding'].apply(pd.Series).add_prefix('embedding_')
        df = pd.concat([emb_df, df], axis=1)

        embedding_prefix = 'embedding_'
        embedding_columns = [col for col in df.columns if col.startswith(embedding_prefix)]

        umap_transformed = self.dim_reduction_pipeline.transform(df.loc[:, embedding_columns])
        n_components = len(umap_transformed[0])
        for i in range(1, n_components + 1):
            df[f'UMAP{i}'] = umap_transformed[:, i - 1]
        #pca_transformed = pca_pl.transform(df.loc[:, embedding_columns])
        #for i in range(1, n_components + 1):
        #    df[f'PC{i}'] = pca_transformed[:, i - 1]

        kmeans_embedding_prefix = 'UMAP'
        embedding_columns = [col for col in df.columns if col.startswith(kmeans_embedding_prefix)]
        cluster_labels = self.kmeans_pipeline.predict(df.loc[:, embedding_columns])
        df['klabels'] = cluster_labels

        embeddings = df[[col for col in df.columns if col.startswith(kmeans_embedding_prefix)]].values.astype('float32')
        self.cluster_index_dict[cluster_labels[0]].add(embeddings)

        os.remove(self.path.cluster_index_dict)
        with open(self.path.cluster_index_dict, 'wb') as file:
            pickle.dump(self.cluster_index_dict, file)

        df = pd.concat([df, self.emb_split_df], axis=0)
        df.reset_index(inplace=True, drop=True)

        os.remove(self.path.df_pickle)
        #print(self.path.df_pickle)
        FilesOps.save_df(df, isCurrentIsRoot)

        self.cluster_index_dict = pickle.load(open(self.path.cluster_index_dict, 'rb'))
        self.emb_split_df = pd.read_pickle(self.path.df_pickle)
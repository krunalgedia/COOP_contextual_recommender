import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.cluster import KMeans
import faiss
import numpy as np
import os
import sys
import joblib
from faiss import write_index, read_index
import pickle
from matplotlib import pyplot as plt
# Add the path to the other directory to the Python path
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../models'))
from src.configs.config import ModelConfig, PathsConfig
from src.utils.preprocessing_ops import PreprocessUtils
from src.utils.file_ops import FilesOps

class Models:
    def __init__(self, dfcls):
        self.df = dfcls.df
        self.path = PathsConfig()

    def check_pca(self, embedding_prefix):
        # Select embedding columns
        embedding_columns = [col for col in self.df.columns if col.startswith(embedding_prefix)]

        # Define the PCA pipeline
        pl = Pipeline(steps=[
            # ('scaler', StandardScaler()),
            ('clf', PCA( random_state=42))
        ])

        pca = pl.fit(self.df.loc[:, embedding_columns])

        # Get explained variance ratio
        explained_variance_ratio = pl.named_steps['clf'].explained_variance_ratio_

        # Calculate cumulative explained variance
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)

        # Find number of components explaining 90% variance
        n_components_90 = np.argmax(cumulative_explained_variance >= 0.90) + 1  # Add 1 to convert from index to count

        # Find number of components explaining 95% variance
        n_components_95 = np.argmax(cumulative_explained_variance >= 0.95) + 1  # Add 1 to convert from index to count

        # Plot explained variance ratio by number of components
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_explained_variance, marker='o', linestyle='-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Explained Variance by Number of Components')
        plt.grid(True)
        plt.axvline(n_components_90, color='red', linestyle='--', label='90% Variance')
        plt.axvline(n_components_95, color='green', linestyle='--', label='95% Variance')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.path.plots_dir, 'pca_explained_var.png'))
        plt.show()

        print("Number of components explaining 90% variance:", n_components_90)
        print("Number of components explaining 95% variance:", n_components_95)

    def apply_pca(self,embedding_prefix, n_components):
        # Select embedding columns
        embedding_columns = [col for col in self.df.columns if col.startswith(embedding_prefix)]

        # Define the PCA pipeline
        pl = Pipeline(steps=[
            # ('scaler', StandardScaler()),
            ('clf', PCA(n_components=n_components, random_state=42))
        ])

        # Apply PCA and create a DataFrame with the transformed columns
        pca_df = pd.DataFrame(pl.fit_transform(self.df.loc[:, embedding_columns]),
                              columns=[f'PC{i}' for i in range(1, n_components + 1)])

        # Concatenate the PCA-transformed columns with the original DataFrame
        self.df = pd.concat([pca_df, self.df], axis=1)

        joblib.dump(pca_df, self.path.pca_joblib)

        FilesOps.save_df(self.df)

        return self.df, pl

    def apply_umap(self, embedding_prefix, n_components):
        # Select embedding columns
        embedding_columns = [col for col in self.df.columns if col.startswith(embedding_prefix)]

        # Define the UMAP pipeline
        pl_umap = Pipeline(steps=[
            ('clf', UMAP(n_components=n_components, random_state=42))
        ])

        # Apply UMAP and create a DataFrame with the transformed columns
        umap_df = pd.DataFrame(pl_umap.fit_transform(self.df.loc[:, embedding_columns]),
                               columns=[f'UMAP{i}' for i in range(1, n_components + 1)])

        # Concatenate the UMAP-transformed columns with the original DataFrame
        self.df = pd.concat([umap_df, self.df], axis=1)

        joblib.dump(pl_umap,self.path.umap_joblib)

        FilesOps.save_df(self.df)

        return self.df, pl_umap

    def cluster_data(self, embedding_prefix, n_clusters):

        # Select embedding columns
        embedding_columns = [col for col in self.df.columns if col.startswith(embedding_prefix)]

        # Define the KMeans pipeline
        plkmeans = Pipeline(steps=[
            # ('scaler', StandardScaler()),
            ('kmeans', KMeans(n_clusters=n_clusters, init='k-means++', random_state=42))
        ])

        # Fit KMeans to the embedding columns
        plkmeans.fit(self.df.loc[:, embedding_columns])

        # Predict cluster labels
        cluster_labels = plkmeans.predict(self.df.loc[:, embedding_columns])

        # Add cluster labels to the DataFrame
        self.df['klabels'] = cluster_labels

        joblib.dump(plkmeans, self.path.kmeans_joblib)
        FilesOps.save_df(self.df)

        return self.df, plkmeans

    def get_centroid_indices(self):

        kmeans_pipeline = joblib.load(self.path.kmeans_joblib)

        # Get the centroids of each cluster
        centroids = kmeans_pipeline.named_steps['kmeans'].cluster_centers_

        # Convert centroids to float32 (required by FAISS)
        centroids = centroids.astype(np.float32)

        # Create a DataFrame with labels and corresponding centroids
        centroids_df = pd.DataFrame({'Centroid': centroids.tolist()})

        # Initialize FAISS index
        index_dim = centroids.shape[1]
        index = faiss.IndexFlatL2(index_dim)
        #index = faiss.IndexFlatIP(index_dim)

        # Add centroids to FAISS index
        index.add(centroids)

        self.index = index

        write_index(index, os.path.join(self.path.index_dir, self.path.centroid_index))

        return index

    def save_faiss_index_by_label(self, embedding_startswith):
        index_dict = {}

        # Get unique labels from the 'klabels' column
        unique_labels = self.df['klabels'].unique()

        # Iterate over unique labels
        for label in unique_labels:
            # Filter the DataFrame for the current label
            filtered_df = self.df[self.df['klabels'] == label]

            # Filter columns with names starting with the specified prefix
            embedding_columns = [col for col in filtered_df.columns if col.startswith(embedding_startswith)]

            # Convert filtered DataFrame to numpy array
            embeddings = filtered_df[embedding_columns].values.astype('float32')

            # Initialize FAISS index
            index = faiss.IndexFlatL2(embeddings.shape[1])  # Assuming the embedding vectors have the same dimension
            #index = faiss.IndexFlatIP(embeddings.shape[1])

            # Add embeddings to the index
            index.add(embeddings)

            # Save the index to the dictionary
            index_dict[label] = index

            self.index_dict = index_dict

            with open(os.path.join(self.path.index_dir, self.path.cluster_index_dict), 'wb') as file:
                pickle.dump(self.index_dict, file)

        return index_dict



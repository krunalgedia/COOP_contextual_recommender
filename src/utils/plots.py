import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#import joblib
import sys
import os
sys.path.append(os.path.abspath('../'))
from src.configs.config import ModelConfig, PathsConfig

class Plotter:
    def __init__(self):
        pass

    @staticmethod
    def plot_pca(df, filename, hue='type'):
        # hue can be klabels

        # Create figure and axes
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        plt.subplots_adjust(right=2.5, top=1.25)

        # Plot PCA components
        sns.scatterplot(x='PC1', y='PC2', data=df, hue=hue, ax=axes)

        # Set title and axis labels with increased font size
        axes.set_title('PCA Plot', fontsize=16)
        axes.set_xlabel('PC1', fontsize=14)
        axes.set_ylabel('PC2', fontsize=14)

        # Increase font size of legend
        axes.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)

        # Increase font size of ticks
        axes.tick_params(axis='both', which='major', labelsize=12)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        path = PathsConfig()
        plt.tight_layout()
        plt.savefig(os.path.join(path.plots_dir, filename))
        #plt.show()

    @staticmethod
    def plot_umap(df, filename, hue='type'):
        # Create figure and axes
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        plt.subplots_adjust(right=2.5, top=1.25)

        # Plot UMAP components
        sns.scatterplot(x='UMAP1', y='UMAP2', data=df, hue=hue, ax=axes)

        # Set title and axis labels with increased font size
        axes.set_title('UMAP Plot', fontsize=16)
        axes.set_xlabel('UMAP1', fontsize=14)
        axes.set_ylabel('UMAP2', fontsize=14)

        # Increase font size of legend
        axes.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)

        # Increase font size of ticks
        axes.tick_params(axis='both', which='major', labelsize=12)

        path = PathsConfig()
        plt.tight_layout()
        plt.savefig(os.path.join(path.plots_dir, filename))
        plt.show()

    @staticmethod
    def cluster_and_plot(min_clusters, max_clusters, embedding_prefix, df, filename):
        embedding_columns = [col for col in df.columns if col.startswith(embedding_prefix)]

        wcss = []
        silhouette_scores = []

        for n_clusters in range(min_clusters, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, max_iter=50)
            kmeans.fit(df.loc[:, embedding_columns])

            # Within-cluster sum of squares
            wcss.append(kmeans.inertia_)

            # Silhouette score
            cluster_labels = kmeans.labels_
            silhouette_avg = silhouette_score(df.loc[:, embedding_columns], cluster_labels)
            silhouette_scores.append(silhouette_avg)
            # print("For n_clusters={}, the silhouette score is {:.4f}".format(n_clusters, silhouette_avg))

        # Plotting
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 5))

        # WCSS plot
        ax1 = sns.lineplot(x=range(min_clusters, max_clusters + 1), y=wcss, label='WCSS', marker='o', color='b')
        ax1.set_ylabel('WCSS', color='b')
        ax1.set_xlabel('Number of Clusters')

        # Silhouette score plot
        ax2 = ax1.twinx()
        sns.lineplot(x=range(min_clusters, max_clusters + 1), y=silhouette_scores, label='Silhouette Coefficient',
                     marker='s', color='r')
        ax2.set_ylabel('Silhouette Coefficient', color='r')

        # Add legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        path = PathsConfig()
        plt.tight_layout()
        plt.savefig(os.path.join(path.plots_dir, filename))

        # Show the plot
        plt.show()

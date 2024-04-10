from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import os

class ModelConfig():
    def __init__(self):
        self.model = SentenceTransformer("all-distilroberta-v1")
        self.rnd = 42

class PathsConfig():
    def __init__(self, isCurrentIsRoot=False):
        self.data_url = 'https://www.dropbox.com/scl/fi/lt7mbtho9d3n9guh9cz2k/coopdata.zip?rlkey=m501w6zggiugila3zps1jfbql&dl=0'
        self.rename_data_zip = 'data.zip'
        self.current_dir = Path.cwd()
        #print('current_dir: ',self.current_dir)
        self.root_dir = self.current_dir.parent
        if isCurrentIsRoot:
            self.root_dir = self.current_dir
        #print('root_dir: ',self.root_dir)
        self.data_dir = self.root_dir.joinpath('data')
        self.models_dir = self.root_dir.joinpath('models')
        self.plots_dir = self.root_dir.joinpath('plots')
        self.index_dir = self.root_dir.joinpath('index')
        self.data_folder = [folder for folder in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, folder))][0]
        self.choco_data = self.data_dir.joinpath(self.data_folder).joinpath('choco.csv')
        self.clean_data = self.data_dir.joinpath(self.data_folder).joinpath('clean.csv')
        self.drink_data = self.data_dir.joinpath(self.data_folder).joinpath('drink.csv')
        self.pca_joblib = self.models_dir.joinpath('pca_pipeline.joblib')
        self.umap_joblib = self.models_dir.joinpath('umap_pipeline.joblib')
        self.final_dim_reduction_pipeline = self.umap_joblib
        self.kmeans_joblib = self.models_dir.joinpath('kmeans_pipeline.joblib')
        self.centroid_index = self.index_dir.joinpath('centroid_index')
        self.cluster_index_dict = self.index_dir.joinpath('cluster_index_dict.pkl')
        self.df_pickle = self.data_dir.joinpath('emb_split_df.pkl')
        self.sample_chocopic = self.plots_dir.joinpath('chocolate.jpeg')
        self.sample_cleanpic = self.plots_dir.joinpath('clean.jpeg')
        self.sample_drinkpic = self.plots_dir.joinpath('drinks.jpeg')
        self.sample_picdict = {'choco': self.sample_chocopic,
                               'drink': self.sample_drinkpic,
                               'clean': self.sample_cleanpic
                               }
        self.sample_cooppic = self.plots_dir.joinpath('coop.png')

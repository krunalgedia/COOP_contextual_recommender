import os
import shutil
import subprocess
import zipfile
import sys
import urllib.request
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import requests
# Add the path to the other directory to the Python path
sys.path.append(os.path.abspath('../'))
from src.configs.config import ModelConfig, PathsConfig


class FilesOps:
    def __init__(self):
        pass

    @staticmethod
    def download_rename_and_unzip(url, new_name):

        path = PathsConfig()
        new_path = os.path.join(path.data_dir, new_name)

        # Download the zip file if not present
        if not os.path.exists(new_path):
            print('Path not found')
            urllib.request.urlretrieve(url, new_name)
            # Ensure the zip file is downloaded successfully
            files = [i for i in os.listdir("./") if 'zip' in i]
            assert len(files) > 0, "Zip file not found after downloading"

            # Move the downloaded file with the new name
            shutil.move(files[0], new_path)

        # Unzip the file
        with zipfile.ZipFile(new_path, 'r') as zip_ref:
            zip_ref.extractall(path.data_dir)

        # Ensure the zip file is extracted successfully
        assert os.path.exists(new_path), "Data directory not found after extraction"

        # Delete the zip file
        #os.remove(new_name)

        '''
            
        #with urlopen(url) as zipresp:
        #    with ZipFile(BytesIO(zipresp.read())) as zfile:
        #        zfile.extractall(path.data_dir)
        #subprocess.run(["wget", url])
        #urllib.request.urlretrieve(url, new_name)

        remote = urllib.request.urlopen(url)  # read remote file
        data = remote.read()  # read from remote file
        remote.close()  # close urllib request
        local = open(new_name, 'wb')  # write binary to local file
        local.write(data)
        local.close()  # close file

        
        # Ensure the zip file is downloaded successfully
        files = [i for i in os.listdir("./") if 'zip' in i]
        assert len(files) > 0, "Zip file not found after downloading"

        path = PathsConfig()
        new_path = os.path.join(path.data_dir,new_name)
        print(new_path)
        # Move the downloaded file with the new name
        shutil.move(files[0], new_path)

        # Unzip the file
        with zipfile.ZipFile(new_path, 'r') as zip_ref:
            zip_ref.extractall(PathsConfig.data_dir)

        # Ensure the zip file is extracted successfully
        assert os.path.exists(new_path), "Data directory not found after extraction"

        # Delete the zip file
        os.remove(new_name)
        '''
    @staticmethod
    def save_df(df, isCurrentIsRoot=False, file_format='pickle'):
        paths = PathsConfig(isCurrentIsRoot)
        filename = paths.df_pickle
        if file_format == 'csv':
            # Save DataFrame to CSV file
            df.to_csv(filename, index=False)
        elif file_format == 'pickle':
            # Save DataFrame to pickle file
            df.to_pickle(filename)
        else:
            raise ValueError("Invalid file format. Choose either 'csv' or 'pickle'.")

        # Ensure the file is saved successfully
        assert os.path.exists(filename), f"File {filename} not found after saving"

        return filename

import sys
import subprocess
import os

def get_script_dir():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()

def install_packages():
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
        requirements_path = os.path.join(script_dir, 'requirements.txt')
        if os.path.exists(requirements_path):
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])
                print("Requirements installed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error installing requirements: {e}")
        else:
            print("requirements.txt not found. Please make sure it is in the root directory.")
    except Exception as e:
        print(f"An error occurred while installing packages: {e}")
        sys.exit(1)

# Get the directory of the current script
script_dir = get_script_dir()

# Install packages before importing other modules
install_packages()


import yaml
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

class PathManager:
    def __init__(self, config):
        self.base_path = script_dir
        self.config = config
        self.paths = self._initialize_paths()

    def _initialize_paths(self):
        paths = {}
        for key, directory in self.config['directories'].items():
            paths[key] = os.path.join(self.base_path, directory)
        return paths

    def get_path(self, key):
        return self.paths.get(key)

    def create_directories(self):
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def run_notebook(notebook_path):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})

    with open(notebook_path, 'w') as f:
        nbformat.write(nb, f)


if __name__ == "__main__":
    config_path = os.path.join(script_dir, 'config.yaml')
    config = load_config(config_path)
    
    path_manager = PathManager(config)
    path_manager.create_directories()

    notebooks = [
        '0_data_articles.ipynb',
        '1_data_description.ipynb',
        '2_data_tickers.ipynb',
        '3_data_embeddings.ipynb',
        '4_kmeans_clustering.ipynb',
        # '5_0_llama_news_parser.ipynb',
        '5_llama_clustering.ipynb'
    ]

    for notebook in notebooks:
        notebook_path = os.path.join(path_manager.get_path('notebooks'), notebook)
        print('\n' + '>'*111 + '\n' + f"Running notebook: {notebook_path}" + '\n' + '>'*111)
        run_notebook(notebook_path)
        print('\n' + '>'*111 + '\n' + f"Finished running: {notebook_path}" + '\n' + '>'*111 + '\n')


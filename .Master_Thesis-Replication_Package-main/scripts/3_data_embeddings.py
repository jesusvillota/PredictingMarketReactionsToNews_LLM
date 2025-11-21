"""
Script converted from notebook: 3_data_embeddings.ipynb
Original notebook: 3_data_embeddings
"""

# <!-- Container with white background -->
# <div style="background-color: white; padding: 20px; border-radius: 10px;">

#   <!-- Name in bold and approximate LaTeX font style with the logo blue color -->
#   <h1 style="font-family: 'Times New Roman', Times, serif; font-weight: bold; color: #38549c; text-align: center;">
#     Jesus Villota Miranda
#   </h1>

#   <!-- Project name in similar style with the logo blue color in italics -->
#   <h2 style="font-family: 'Times New Roman', Times, serif; color: #38549c; text-align: center; font-style: italic;">
#     Predicting Market Reactions to News: An LLM-Based Approach Using Spanish Business Articles
#   </h2>

#   <!-- CEMFI logo centered -->
#   <div style="text-align: center; margin-bottom: 40px;">
#     <img src="https://www.cemfi.es/images/Logo-Azul.png" alt="CEMFI Logo" style="width:200px;">
#   </div>

#   <!-- Catchy message about authorship -->
#   <p style="font-family: 'Times New Roman', Times, serif; color: #38549c; text-align: center; font-size: 1.2em;">
#     All code and work associated with this project are solely created and authored by Jesus Villota Miranda. © 2024
#   </p>

#   <!-- Contact information with logos -->
#   <p style="font-family: 'Times New Roman', Times, serif; color: #38549c; text-align: center; font-size: 1em;">
#     Contact:
#     <a href="mailto:jesus.villota@cemfi.edu.es" style="color: #38549c;">
#       <img src="https://www.logolynx.com/images/logolynx/64/64319177556c729f1806922bcd3adef5.png" alt="Email Logo" style="width: 20px; vertical-align: middle;">
#       jesus.villota@cemfi.edu.es
#     </a> |
#     <a href="https://www.linkedin.com/in/jesusvillotamiranda/" target="_blank" style="color: #38549c;">
#       <img src="https://1.bp.blogspot.com/-onvhHUdW1Us/YI52e9j4eKI/AAAAAAAAE4c/6s9wzOpIDYcAo4YmTX1Qg51OlwMFmilFACLcBGAsYHQ/s1600/Logo%2BLinkedin.png" alt="LinkedIn Logo" style="width: 20px; vertical-align: middle;">
#       LinkedIn
#     </a>
#   </p>

# </div>


# Preamble

# Packages

# %reset -f  # Jupyter magic command removed
import csv
import os
import pandas as pd
import numpy as np
import ast
import bisect

from datetime import datetime

import torch
from sentence_transformers import SentenceTransformer

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

from joblib import Parallel, delayed
import multiprocessing

# Directories

import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def create_project_structure(base_path, directories):
    for _, directory in directories.items():
        dir_path = os.path.join(base_path, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

if __name__ == "__main__":
    base_path = os.path.dirname(os.getcwd())
    config_path = os.path.join(base_path, 'config.yaml')
    config = load_config(config_path)
    create_project_structure(base_path, config['directories'])

path_processed_data = os.path.join(base_path, config['directories']['processed_data'])

# Functions

R = pd.read_csv(os.path.join(path_processed_data, 'R_KMeans.csv'), index_col=0, parse_dates=True)
trading_days = [dt.date() for dt in R.index]
R.index = trading_days

class TradingCalendarAdjustments:
    def __init__(self, trading_days, closing_time='17:30:00'):
        self.trading_days = sorted(trading_days)  # Ensure the trading days are sorted
        self.closing_time = datetime.strptime(closing_time, '%H:%M:%S').time()
    def is_trading_day(self, date):
        return date in self.trading_days

    def closest_trading_day_at_or_before(self, day_x):
        index = bisect.bisect_right(self.trading_days, day_x) - 1
        if index < 0:
            return None
        return self.trading_days[index]
    
    def closest_trading_day_at_or_after(self, day_x):
        index = bisect.bisect_left(self.trading_days, day_x)
        if index == len(self.trading_days):
            return None
        return self.trading_days[index]

    def next_trading_day(self, date):
        date_corrected = self.closest_trading_day_at_or_before(date)
        if date_corrected is None:
            return None
        index = self.trading_days.index(date_corrected) + 1
        if index >= len(self.trading_days):
            return None
        return self.trading_days[index]

    def impute_date_affect(self, publ_datetime):
        publ_date = publ_datetime.date()
        publ_time = publ_datetime.time()
        if self.is_trading_day(publ_date) and publ_time < self.closing_time:
            return publ_date
        else:
            return self.next_trading_day(publ_date)


adj = TradingCalendarAdjustments(trading_days)

# Data

# Include column with "effective treamment date": $\tilde d_0^i$

# $$\tilde{d}_0^i:=\left\{\begin{array}{lll}d_0^i & \text { if } & d_0^i \in \tilde{\mathfrak{d}} \wedge t_0^i<17: 30: 00.000 \\ \Lambda\left(d_0^i\right) & \text { if } & d_0^i \notin \tilde{\mathfrak{d}} \vee t_0^i>17: 30: 00.000\end{array}\right.$$

# where $\Lambda(d):=\min \{\tilde{d} \in \tilde{\mathfrak{d}} \mid \tilde{d} \geq d\}$

D = pd.read_csv(os.path.join(path_processed_data, 'D.csv'))
D['publ_datetime'] = pd.to_datetime(D['publ_datetime'])
D['date_affect'] = D['publ_datetime'].apply(adj.impute_date_affect)
D['tickers'] = D['tickers'].apply(lambda x: ast.literal_eval(x)) # Convert the 'tickers' column from string representation of lists to actual lists

D

# 
# ---
# ---

# Embeddings

# Get the vector embedding representation associated to each article $i\in\mathcal D$. 

# $$\text{News Article $i$} \longrightarrow \mathbf{e}^i \in \mathbb{R}^{512}$$

# Defining the models and the function to get embeddings

model_dict = {
    'paraphrase-MiniLM-L6-v2': SentenceTransformer('paraphrase-MiniLM-L6-v2'),
    'paraphrase-multilingual-MiniLM-L12-v2': SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'),
    'distiluse-base-multilingual-cased-v1': SentenceTransformer('distiluse-base-multilingual-cased-v1'),
}

# Function to get embeddings for a single article based on the chosen model
def get_embedding(article, model_name):
    if model_name in model_dict:
        model = model_dict[model_name]
        return model.encode(article).tolist()

# Getting the embeddings for all articles

available_models = list(model_dict.keys())
print(f"Available models: {available_models}")

selected_model = 'distiluse-base-multilingual-cased-v1'
print(f"Selected model: {selected_model}")

# Apply the get_embedding function to each article using the selected model
print("Calculating embeddings...")
D['embeddings'] = D['articles'].apply(lambda x: get_embedding(x, selected_model))
print("Done.")

# Now D contains a new column 'embeddings' with the vector representations of each article
D.head()

# Save the data witht the embeddings and the date_affect column
D.to_csv(os.path.join(path_processed_data, 'D_embeddings.csv'))

# 
# ---
# ---

"""
Script converted from notebook: 0_data_articles.ipynb
Original notebook: 0_data_articles
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
import os
import csv
import pandas as pd
import numpy as np
from datetime import datetime
import re

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

path_raw_data       = os.path.join(base_path, config['directories']['raw_data'])
path_processed_data = os.path.join(base_path, config['directories']['processed_data'])

# Functions

def eliminate_text_after_word(text, word_x):
    index = text.find(word_x)
    if index != -1:
        return text[:index]
    else:
        return text

def extract_datetime(text):
    datetime_pattern = r'\d{2}-\d{2}-\d{2} \d{4}GMT' # regex pattern for datetime
    datetime_match = re.search(datetime_pattern, text) # search for the pattern in the text
    if datetime_match:
        datetime_str = datetime_match.group(0) # extract the datetime string
        return datetime_str
    else:
        return None
    
def convert_to_datetime(timestamp_ms):
    # Convert milliseconds to seconds by dividing by 1000
    timestamp_seconds = timestamp_ms / 1000
    # Convert timestamp to a datetime object
    date_time = datetime.fromtimestamp(timestamp_seconds)
    return date_time

def extract_tickers_from_article(article):
    # Define the pattern to match '(WHATEVER.MC)' where WHATEVER is any upper-case ticker symbol
    pattern = r'\(([A-Z]+\.MC)\)'
    # Find all matches of the pattern in the article
    matches = re.findall(pattern, article)
    # Return unique instances of matched ticker symbols
    unique_tickers = list(set(matches))
    return unique_tickers


# 
# ---
# ---

# Read Raw Data

# Read the raw data
df_full = pd.read_parquet(os.path.join(path_raw_data, 'ibex_sample.pqt.gziq'))

# Convert the publication_datetime from EPOCH to YYYY-MM-DD
df_full['publication_datetime'] = pd.to_datetime(df_full['publication_datetime'], unit='ms').dt.strftime('%Y-%m-%d')

# Sort by publication_datetime
df_full = df_full.sort_values('publication_date') 

# Extracting a reduced subset of the data

# Filter the columns we need
columns_to_keep1 = ['publication_date', 'title', 'snippet', 'body', 'word_count', 'company_codes_about', 'company_codes_about_ticker_exchange'] # columns of interest
df = df_full[columns_to_keep1].copy() # keep only the columns we need

# Obtain the publication_date
df['publication_datetime'] = df['publication_date'].apply(convert_to_datetime)
df.drop(columns=['publication_date'], inplace=True)
df['publication_date'] = df_full['publication_datetime']

# 
# ---
# ---

# Data Cleaning

# Filtering articles that are **NOT** referred to firms (disregard news about political and economic agenda)

df_filtered = df[(df['company_codes_about'] != '') & (df['title'] != 'España: Agenda política y económica -Semana')].copy() # filter out articles that are not referred to firms

publ_datetime = df_filtered['publication_datetime'].copy().tolist()

# Merging the Title, Snippet & Body

#### Filter only [publication_date, title, snippet, body]
columns_to_keep2 = ['title', 'snippet', 'body']
df1 = df_filtered[columns_to_keep2].copy()
df1.fillna('', inplace=True)
Documents = df1['title'] + '. ' + df1['snippet'] + '. ' + df1['body']
Documents = pd.DataFrame({'publ_datetime': publ_datetime, 'articles': Documents})

Documents

# Cleaning `Documents`

# * Eliminate all the words that appear after `["-Escriba a", "Editado por", "(END)", "Versión española de", "Escriba a"]`
# * Eliminate redundant expressions that don't add information `['MARKET TALK: ', '(EFE Dow Jones)--' , '(EFE Dow Jones).-']`
# * Eliminate email directions and "Redactada por [Nombre] [Apellido]"

DocsFiltered = Documents.copy()

#### CLEANING AFTER TEXT ####
eliminate_text_after_these_words = ["-Escriba a", "Editado por", "(END)", "Versión española de", "Escriba a", "Traductores:"]
for word in eliminate_text_after_these_words:
    Document_clean = DocsFiltered['articles'].apply(eliminate_text_after_word, word_x=word)
    DocsFiltered['articles'] = Document_clean

#### CLEANING SPECIFIC EXPRESSIONS ####
expressions_to_remove = ['MARKET TALK: ', 'MADRID', 'BARCELONA', 'LONDRES', 'MÉXICO', 'ROMA', 'BRUSELAS', 'FRÁNCFORT', 'SÍDNEY' , 'PARÍS', 'RÍO DE JANEIRO',
                         '(EFE Dow Jones)--' , '(EFE Dow Jones).-', '(EFE Dow Jones)', '(MORE TO FOLLOW)', 'Dow Jones Newswires', 'GMT', 'gmt', 'Gmt', 
                         '(rodrigo.demiguelroncal@dowjones.com )', '(Reenfoca titular y añade detalles a lo largo del texto)', 'Rodríguez',
                         '--Giulia Petroni contribuyó a esta nota', '--Mauro Orrù contribuyó a esta nota', '-Ben Otto contribuyó a esta nota']
for expression in expressions_to_remove:
    Document_clean = DocsFiltered['articles'].str.replace(expression, '')
    DocsFiltered['articles'] = Document_clean

#### CLEANING TEXT PATTERNS ####
patterns = [
    r'\([^)]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\)', # eliminates (name@example.domain)
    r'\(\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\b\)', # eliminates (name@example)
    r'\nRedactada por [A-Za-záéíóúÁÉÍÓÚ]+\s[A-Za-záéíóúÁÉÍÓÚ]+', # eliminates Redactada por Name Surname
    r'Redactada por [A-Za-záéíóúÁÉÍÓÚ]+\s[A-Za-záéíóúÁÉÍÓÚ]\sy\s[A-Za-záéíóúÁÉÍÓÚ]+\s[A-Za-záéíóúÁÉÍÓÚ]' # eliminates Redactada por Name Surname y Name Surname
    r'\([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}; @\w+\)', # eliminates (email; @twitter)
    r'Por\s[A-Za-záéíóúÁÉÍÓÚ]+\s[A-Za-záéíóúÁÉÍÓÚ]+\s[A-Za-záéíóúÁÉÍÓÚ]+' # eliminates Por Name1 Name2 Name3
]

for pattern in patterns:
    Document_clean = DocsFiltered['articles'].apply(lambda x: re.sub(pattern, '', x))
    DocsFiltered['articles'] = Document_clean


DocsFiltered

# Filter the articles that explicitly mention stock market tickers of publicly traded Spanish firms (regex pattern: TICKER.MC)

DocsFiltered['tickers'] = DocsFiltered['articles'].apply(extract_tickers_from_article)
Docs_n_Tickers = DocsFiltered[DocsFiltered['tickers'].apply(len) > 0].copy() # drop articles with no tickers


Docs_n_Tickers

# Save to CSV

Docs_n_Tickers.to_csv(os.path.join(path_processed_data, 'D.csv'), index=False)

# 
# ---
# ---

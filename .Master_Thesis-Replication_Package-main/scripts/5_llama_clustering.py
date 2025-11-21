"""
Script converted from notebook: 5_llama_clustering.ipynb
Original notebook: 5_llama_clustering
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


# <h1 style="font-family: 'Times New Roman', Times, serif; font-weight: bold; color: #38549c; text-align: center;">
#     kernel: Python 3.9.6
#   </h1>

# Preamble

# Packages

# %reset -f  # Jupyter magic command removed

model = 'LLAMA'

import csv
import os
import pandas as pd
import numpy as np
import ast
from datetime import timedelta
import itertools

import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import statsmodels.api as sm
import scipy.stats

from joblib import Parallel, delayed
from tabnanny import verbose
from collections import OrderedDict

# Plotting packages
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.dates import DateFormatter
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

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
path_output         = os.path.join(base_path, config['directories']['output_llama'])

# Plotting Parameters

TITLE_FONTSIZE  = 18
LABEL_FONTSIZE  = 20
TICK_FONTSIZE   = 18
LEGEND_FONTSIZE = 16

TITLE_PAD       = 20
LABEL_PAD       = 20

# 
# ---
# ---

# Data

# Load the articles parsed by LLaMA-3. 

# $$\mathcal{B}_{L L M}:=\left\{(i, j) \mid i \in \mathcal{D} \wedge j \in \mathcal{F}_{L L M}^i\right\}$$

B = pd.read_csv(os.path.join(path_raw_data, "LLAMA_parsed_news.csv"))
B

# Load the return data: 

# $$
# ~\mathcal R:= \left\{\{r^f_d\}_{d\in\tilde{\mathfrak{d}}}, \{r^M_d\}_{d\in\tilde{\mathfrak{d}}}, \{\{\{r^j_d\}_{d\in\tilde{\mathfrak{d}}}\}_{j\in \mathcal F^i}\}_{i\in\mathcal D}\right\}
# $$

R = pd.read_csv(os.path.join(path_processed_data, 'R_LLAMA.csv'), index_col=0, parse_dates=True)
𝖉 = [dt.date() for dt in R.index]
R.index = 𝖉

# Read successful_tickers 
with open(os.path.join(path_processed_data, 'successful_tickers_LLAMA.txt'), 'r') as f:
    successful_tickers = f.read().splitlines()

# Read failed_tickers
with open(os.path.join(path_processed_data, 'failed_tickers_LLAMA.txt'), 'r') as f:
    failed_tickers = f.read().splitlines()

R

# Load the D dataset to assign date_affect to B
D = pd.read_csv(os.path.join(path_processed_data, 'D_embeddings.csv'), index_col=0)
# only keep 'articles' and 'date_affect' columns
D = D[['articles', 'date_affect']]

D

B = B.merge(D, left_on='article_id', right_index=True, how='left')
# ensuring the right format for date_affect
if not pd.api.types.is_datetime64_any_dtype(B['date_affect']):
    B['date_affect'] = pd.to_datetime(B['date_affect']).dt.date
del D
B

# 
# ---
# ---

# Clustering

# We can define the set $\mathcal{B}_{L L M}:=\left\{(i, j) \mid i \in \mathcal{D} \wedge j \in \mathcal{F}_{L L M}^i\right\}$ and ask the LLM to classify the event or shock described in the news article for each firm $j \in \mathcal{F}_{L L M}^i$. This classification involves assigning each pair $(i, j) \in \mathcal{B}_{L L M}$ with a choice from the following sets:
# $$
# \begin{array}{ll}
# \text { "shock type" } & \left.\mathcal{S}_T=\text { \{ demand, supply, financial, technological, policy }\right\} \\
# \text { "shock magnitude" } & \left.\mathcal{S}_M=\text {\{ minor, major }\right\} \\
# \text { "shock direction" } & \left.\mathcal{S}_D=\text { \{ positive, negative }\right\}
# \end{array}
# $$

# The clustering of news articles follows naturally by taking the Cartesian product of these three sets: $\mathcal{G}_{L L M}=\mathcal{S}_T \times \mathcal{S}_M \times \mathcal{S}_D$, and the total number of cluster is now $k_{L L M}=\left|\mathcal{G}_{L L M}\right|=20$. Consequently, a news article to which the LLM assigns $s_T \in \mathcal{S}_T, s_M \in \mathcal{S}_M, s_D \in \mathcal{S}_D$ will belong to cluster $\left(s_T, s_M, s_D\right) \in \mathcal{G}_{L L M}$. Formally, the set of all possible clusters is defined as:
# $$
# \mathcal{G}_{L L M}=\left\{\left(s_T, s_M, s_D\right) \mid s_T \in \mathcal{S}_T, s_M \in \mathcal{S}_M, s_D \in \mathcal{S}_D\right\}
# $$

# Then, we can map $\mathcal{G}_{L L M} \rightarrow\left\{k \in \mathbb{N}_0 \mid 0 \leq k \leq 19\right\}$ to obtain a numerical representation of the clusters. 

# Define the sets
shock_type      = ['demand', 'supply', 'financial', 'technology', 'policy']
shock_magnitude = ['minor', 'major']
shock_direction = ['positive', 'negative']

# Create the Cartesian product of the sets
clusters = list(itertools.product(shock_type, shock_magnitude, shock_direction))

# Map each combination to a unique number
cluster_map = {cluster: idx for idx, cluster in enumerate(clusters)}

# Function to get cluster number for a row
def get_cluster(row):
    return cluster_map[(row['shock_type'], row['shock_magnitude'], row['shock_direction'])]

# Apply the function to assign clusters
B['cluster'] = B.apply(get_cluster, axis=1)
B

# 
# ---
# ---

# Sample of articles from each cluster

# Set the sample size
sample_size = 20

# Create the dictionary to store sampled articles
sampled_articles_dict = {}

# Group by cluster and sample 20 articles per cluster, focusing on the wording of the articles
for cluster, group in B.groupby('cluster'):
    sampled_articles_dict[cluster] = group['articles'].sample(min(len(group), sample_size)).tolist()

# Print the dictionary with sampled articles, separating each article with a delimiter
for cluster, articles in sampled_articles_dict.items():
    if cluster == 19:
        print(f"Cluster {cluster}:")
        for article in articles:
            print(article)
            print('_' * 50)  # Delimiter to separate articles
        print("\n")


# 
# ---
# ---

# Data Split

split1 = 0.8
split2 = 0.6

split_threshold1 = int(len(B['article_id'].unique()) * split1)
split_threshold2 = int(len(B['article_id'].unique()) * split1 * split2)

# if article_id > split_threshold, then "Split" column is "Test", if split_threshold2 <= article_id <= split_threshold, then "Split" column is "Validation", else "Train"
B['split'] = np.where(B['article_id'] > split_threshold1, 'Test', np.where(B['article_id'] > split_threshold2, 'Validation', 'Train'))
B

# 
# ---
# ---

# Distribution of Articles through Clusters

cluster_counts = B['cluster'].value_counts().sort_index()
print(cluster_counts)

save_this_plot  = True
show_title      = True
plot_density    = True

#===============================================================================================================

# Calculate the distribution of articles per cluster
cluster_counts = B['cluster'].value_counts().sort_index()

# Create a bar plot for the counts
fig, ax1 = plt.subplots(figsize=(12, 6))
cluster_counts.plot(kind='bar', ax=ax1, alpha=0.6, color='blue', edgecolor='black')
ax1.set_xlabel('Cluster', fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
ax1.set_ylabel('Number of Articles', fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
ax1.set_title('Distribution of Articles per Cluster $~\mid~$ All data', fontsize=TITLE_FONTSIZE, pad=TITLE_PAD) if show_title else None
ax1.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
ax1.grid(True, alpha=0.3, linestyle=':')
ax1.set_facecolor('#f5f5f5')

ax1.spines['top'].set_visible(False)


ax1.spines['right'].set_visible(False) if not plot_density else None

if plot_density: 
    # Create a secondary y-axis for the density plot
    ax2 = ax1.twinx()
    sns.kdeplot(B['cluster'], ax=ax2, color='red', linewidth=2)
    ax2.set_ylabel('Density', fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax2.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax1.set_facecolor('#f5f5f5')
    ax2.spines['top'].set_visible(False)

# Save the plot if requested
plt.savefig(os.path.join(path_output, f'{model}_Cluster_Distribution.pdf'), bbox_inches='tight') if save_this_plot else None
plt.show()


save_this_plot  = True
show_title      = True
plot_density    = True
#===============================================================================================================

for split in B['split'].unique():
    # Filter the data for the current split
    split_data = B[B['split'] == split]
    
    # Calculate the distribution of articles per cluster
    cluster_counts = split_data['cluster'].value_counts().sort_index()
    
    # Create a bar plot for the counts
    fig, ax1 = plt.subplots(figsize=(12, 6))
    cluster_counts.plot(kind='bar', ax=ax1, alpha=0.6, color='blue', edgecolor='black')
    ax1.set_xlabel('', fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)  # Set empty x-label
    ax1.set_ylabel('', fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)  # Set empty y-label
    ax1.set_title(f'Distribution of Articles per Cluster $~|~$ Split: {split}', fontsize=TITLE_FONTSIZE, pad=TITLE_PAD) if show_title else None
    
    ax1.tick_params(axis='y', which='major', labelsize=TICK_FONTSIZE)  # Set y-ticks size for the left axis
    ax1.tick_params(axis='x', which='major', labelsize=TICK_FONTSIZE)  # Set x-ticks size
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_facecolor('#f5f5f5')
    ax1.spines['top'].set_visible(False)

    if plot_density: 
        # Create a secondary y-axis for the density plot
        ax2 = ax1.twinx()
        sns.kdeplot(split_data['cluster'], ax=ax2, color='red', linewidth=2)
        ax2.set_ylabel('', fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)  # Set empty y-label for density plot
        ax2.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)  # Remove y-ticks for the right axis
        ax2.tick_params(axis='x', which='major', labelsize=TICK_FONTSIZE)  # Ensure x-ticks size matches the left axis
        ax1.set_facecolor('#f5f5f5')
        ax2.spines['top'].set_visible(False)

    # Save the plot if requested
    if save_this_plot:
        plt.savefig(os.path.join(path_output, f'{model}_Cluster_Distribution_{split}.pdf'), bbox_inches='tight')
    
    plt.show()

# 
# ---
# ---

# Trading Strategy Data

# <span style="color:orange"> 1) Fit the Market Model </span>
# $$r_d^j=\alpha^{(i, j)}+\beta^{(i, j)} r_d^M+\epsilon_d^{(i, j)} \quad \forall d \in \mathcal{M}^i$$

# *where the market model window is defined as $\mathcal{M}^i:=\{d \in \tilde{\mathfrak{d}} \mid \mathbb{D}_{\tilde{\mathfrak{d}}}(\mathbb{I}_{\tilde{\mathfrak{d}}}(\tilde{d}_0^i)-w_b-w_m) \leq d \leq \mathbb{D}_{\tilde{\mathfrak{d}}}(\mathbb{I}_{\tilde{\mathfrak{d}}}(\tilde{d}_0^i)-w_b)\}$, with a buffer of $w_b=10$ trading days before the effective treatment date, and a market model window length of $w_m=100$ trading days.*


# <span style="color:orange"> 2) Compute the returns of the Hedged Portfolio (Abnormal Returns, `AR`) over the $L$ trading days </span>

# $$A R_d^{(i, j)} := r_d^j-\beta^{(i, j)} r_d^M=\alpha^{(i, j)}+\epsilon_d^{(i, j)}$$



# <span style="color:orange"> 3) Compute the Cumulative Abnormal Return (`CAR`) over the $L$ trading days  </span>

# $$
# {C A R}^{(i, j)}=\prod_{d\in\mathcal{M}^i}\left(1+A R_d^{(i, j)}\right) - 1
# $$

# <span style="color:orange"> 4) Compute the Average Daily Return (`μ`), Standard Deviation (`σ`), and Sharpe Ratio (`SR`) over the $L$ trading days  </span>
# \begin{array}{ll}
# \mu_L^{(i, j)} 
# &=
# \frac{1}{L+1} \sum_{d\in\mathcal{M}^i} \ln \left(1+A R_d^{(i, j)}\right)
# \\[1.5em]
# \sigma_L^{(i, j)}
# &=
# \sqrt{\frac{1}{L} \sum_{d\in\mathcal{M}^i}\left[\ln \left(1+A R_d^{(i, j)}\right)-\mu_L^{(i, j)}\right]^2}
# \\[1.5em]
# S R_L^{(i, j)}
# &=
# \mu_L^{(i, j)} / \sigma_L^{(i, j)}
# \end{array}

def TradingStrategy_Data(ticker, date_affect, R=R, successful_tickers=successful_tickers, L_max=260, MarketModel_window=100, MarketModel_buffer=10):
    """
    Calculate the Cumulative Abnormal Return (CAR) for a given stock ticker and event date.

    Parameters:
    ticker (str): The stock ticker symbol.
    date_affect (datetime.date): The date of the event affecting the stock.
    R (pd.DataFrame): DataFrame containing the returns data.
    successful_tickers (list): List of successful tickers.
    L_max (int, optional): The holding period. Default is 260.
    MarketModel_window (int, optional): The window size for the market model estimation. Default is 100.
    MarketModel_buffer (int, optional): The buffer period before the event date for the market model. Default is 10.

    Returns:
    tuple: The Cumulative Abnormal Return (CAR), average daily return (μ), annualized return, standard deviation (σ), Sharpe Ratio (SR), or (np.nan, np.nan, np.nan, np.nan, np.nan) if an error occurs.
    """
    if ticker not in successful_tickers:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    try:
        # Ensure date_affect is in the index
        if date_affect not in R.index:
            raise ValueError("date_affect not in DataFrame index")

        # We work with indices to ensure trading days
        idx_date_affect        = R.index.get_loc(date_affect)
        idx_Start_MarketModel  = idx_date_affect - MarketModel_window - MarketModel_buffer
        idx_End_MarketModel    = idx_date_affect - MarketModel_buffer 

        # Ensure indices are within bounds
        if idx_Start_MarketModel < 0 or idx_End_MarketModel > len(R):
            raise IndexError("Market model window indices out of bounds")

        # 1) Fit the Market Model
        y = R.iloc[idx_Start_MarketModel:idx_End_MarketModel + 1, R.columns.get_loc(f'r_{ticker}_excess')]
        X = sm.add_constant(R.iloc[idx_Start_MarketModel:idx_End_MarketModel + 1, R.columns.get_loc('r_market_excess')])
        model = sm.OLS(y, X, missing='drop').fit()

        # 2) Compute the returns of the Hedged Portfolio over the L trading days: r_t^i - beta r_t^M = alpha + epsilon_t =: AR
        alpha = model.params['const']
        epsilon_t = R[f'r_{ticker}_excess'] - model.predict(sm.add_constant(R['r_market_excess']))  # Computes the time series of residuals for all dates: [In-Sample for the Market Model Window] & [Out-of-Sample for the Rest]
        AR_vector = alpha + epsilon_t

        save = []

        for L in range(0, L_max+1):

            # Abnormal Returns at day date_affect + L
            AR = AR_vector.iloc[idx_date_affect + L]  # Abnormal Return

            # 3) Compute the Cumulative Abnormal Return (CAR) over the L trading days 
            CAR = (1 + AR_vector.iloc[idx_date_affect : idx_date_affect + L + 1]).prod() - 1  # Cumulative Abnormal Return : it is the accumulation of the daily abnormal returns over the L trading days
            
            # 4) Compute the Average Daily Return (μ), Standard Deviation (σ), and Sharpe Ratio (SR) over the L trading days 
            μ, σ, SR = np.nan, np.nan, np.nan
            if L > 0:
                AR_window = AR_vector.iloc[idx_date_affect : idx_date_affect + L + 1]
                μ   = (1 / (L + 1)) * np.sum(np.log(1 + AR_window))                 # Average daily return
                σ   = np.sqrt((1 / L) * np.sum((np.log(1 + AR_window) - μ) ** 2))  # Standard deviation of the residuals
                SR  = (μ / σ) * np.sqrt(252) if σ != 0 else np.nan         # Sharpe Ratio

            save.append((AR, CAR, μ, σ, SR))

        # Convert the list of tuples to a DataFrame
        TS_data = pd.DataFrame(save, columns=['AR', 'CAR', 'μ', 'σ', 'SR'])

        return TS_data

    except Exception as e:
        print(f'Error for {ticker} on {date_affect}: {e}')
        return None
        # return df

# - The code below runs the procedure using **parallelization**


# - Some errors will appear as there are some tickers for which the price data is incomplete or missing (delisted tickers)
#     - **Ignore** these type of errors: "*Error for [TICKER] on [DATE]: zero-size array to reduction operation maximum which has no identity*"

# - Running this codes takes $\approx 25$ seconds

def process_row(row_idx, row):
    ticker      = row['tickers']
    date_affect = row['date_affect']
    try:
        TS_data = TradingStrategy_Data(ticker, date_affect)
        return row_idx, TS_data
    except Exception as e:
        print(f'Error for {ticker} on {date_affect}: {e}')
        return row_idx, None

# Use joblib to parallelize the computation
results = Parallel(n_jobs=-1)(delayed(process_row)(row_idx, row) for row_idx, row in B.iterrows())

# Save the results in the dictionary
TS_dict = {row_idx: TS_data for row_idx, TS_data in results}

# 
# ---
# ---

# Average CARs across clusters for each split

CAR_split_cluster = {} # Dictionary to store the cumulative abnormal returns (CAR) for each split and cluster

for row_idx, row in B.iterrows():
    TS_data = TS_dict[row_idx]
    if row_idx in TS_dict and TS_data is not None and isinstance(TS_data, pd.DataFrame):
        split    = row['split']         # Split of the row
        cluster  = row['cluster']       # Cluster of the row
        CAR      = TS_data['CAR'] + 1   # Vector of Cumulative Abnormal Returns (CAR) 

        # Create dictionary keys if they don't exist
        if (split, cluster) not in CAR_split_cluster:
            CAR_split_cluster[(split, cluster)] = {'CAR_sum': 0, 'count': 0}

        # Accumulate CAR values and count
        CAR_split_cluster[(split, cluster)]['CAR_sum'] += CAR  # Accumulate the CAR values
        CAR_split_cluster[(split, cluster)]['count']   += 1    # Increment the count

CAR_split_cluster = OrderedDict(sorted(CAR_split_cluster.items(), key=lambda x: x[0][1]))       # Sort the dictionary by cluster number (ensures that the clusters are in order, it looks better this way to have an ordered legend when plotting)

#=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

CAR_average = {} # Dictionary to store the average cumulative abnormal returns (CAR) for each split and cluster

for (split, cluster), values in CAR_split_cluster.items():
    if values['count'] > 0:
        CAR_average[(split, cluster)] = values['CAR_sum'] / values['count']     # Compute the average CAR
    else:
        CAR_average[(split, cluster)] = float('nan')                            # or 0 if you prefer

# Print the average SR for each cluster and split
for key, avg_CAR in CAR_average.items():
    print(f"Split: {key[0]}, Cluster: {key[1]}, Average CAR: {avg_CAR}")

# Plot the time series of average CARs for each cluster (in separate plots for each split)

save_this_plot  = True
show_title      = True
#===============================================================================================================
for split_plot in B['split'].unique():
    fig, ax = plt.subplots(figsize=(12, 6))
    for (split,cluster), avg_CAR in CAR_average.items():
        if len(avg_CAR) > 0 and split == split_plot:
            avg_CAR_subset = avg_CAR[:100]
            ax.plot(avg_CAR_subset, label=f'Cluster {cluster}')
    ax.set_title(f'Average CARs for each cluster ({split_plot})', fontsize=TITLE_FONTSIZE, pad=TITLE_PAD) if show_title else None
    ax.set_xlabel('Trading Days', fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
    ax.set_ylabel('Average CAR', fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    ax.grid(True)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=7)
    plt.savefig(os.path.join(path_output, f"{model}_CARs_by_cluster_[{split_plot}].pdf"), bbox_inches='tight') if save_this_plot else None
    plt.show()
#===============================================================================================================

# 
# ---
# ---

# Hyperparameters

# $$\begin{array}{ll}
# L &= 4 \\
# \theta &= \lfloor 0.5 \cdot k \rfloor = 9
# \end{array}$$

# Note that $k=19$ in the train and validation samples due to the fact that there is one cluster for which there are no assignments

#============================== HYPERPARAMETERS ==============================#
L = 4                      # Holding period of the positions
k_opt = len(B['cluster'].unique()) 
prop_of_k = 0.5
θ = int(prop_of_k*k_opt)    # Number of Traded clusters
#=============================================================================#
print(f"-  Holding Period: {L}")
print(f"-  Number of Traded Clusters: {θ}")

# 
# ---
# ---

# Average Sharpe Ratio across clusters for each split

# $$
# \overline{S R}_g =\frac{1}{\left|\mathcal{B}_g \right|} \sum_{(i, j) \in \mathcal{B}_g} S R_L^{(i, j)}
# $$
# where: $\mathcal{B}_g:= \{(i, j) \mid (i, j) \in \mathcal{B} \wedge i \in \mathcal{D}_g\}$.

SR_split_cluster_dict = {}      # Initialize dictionary to store sum of SR and count of observations for each cluster and split

for idx_row, row in B.iterrows():
    TS_data = TS_dict[idx_row]
    if TS_data is not None and isinstance(TS_data, pd.DataFrame) and len(TS_data) > L:
        split   = row['split']
        cluster = row['cluster']
        SR_L    = TS_data['SR'][L]

        # Create dictionary keys if they don't exist
        if (split, cluster) not in SR_split_cluster_dict:
            SR_split_cluster_dict[(split, cluster)] = {'SR_sum': 0, 'count': 0}

        # Accumulate SR values and count
        SR_split_cluster_dict[(split, cluster)]['SR_sum'] += SR_L # Accumulate the SR values
        SR_split_cluster_dict[(split, cluster)]['count']  += 1    # Increment the count


SR_split_cluster_dict = OrderedDict(sorted(SR_split_cluster_dict.items(), key=lambda x: x[0][1]))       # Sort the dictionary by cluster number (ensures that the clusters are in order, it looks better this way to have an ordered legend when plotting)

#=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=

SR_Average_dict = {}        # Initialize dictionary to store average SR for each cluster and split

for (split, cluster), values in SR_split_cluster_dict.items():
    if values['count'] > 0:
        SR_Average_dict[(split, cluster)] = values['SR_sum'] / values['count']  # Compute the average SR
    else:
        SR_Average_dict[(split, cluster)] = float('nan')  # or 0 if you prefer

#=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=


print(f"{'Split':<20} {'Cluster':<10} {'Average SR':<10}")
print('-' * 40)

previous_cluster = None # Initialize a variable to keep track of the previous cluster

for idx, (key, avg_sr) in enumerate(SR_Average_dict.items(), start=1):
    current_cluster = key[1]
    if previous_cluster is not None and current_cluster != previous_cluster:
        print('-' * 40)
    print(f"{key[0]:<20} {key[1]:<10} {avg_sr:<10.6f}")
    previous_cluster = current_cluster
    
# # del SR_split_cluster_dict, idx_row, row, TS_data, split, cluster, SR_L, values, key, avg_sr

# Ranking of average Sharpe Ratios

#  $$\mathfrak{R}_g^{split}=\sum_{h \in \mathcal{G}} \mathbf{1}\left(\overline{S R}_h^{split} \geq \overline{S R}_g^{split}\right)$$

SR_split_dict = {} # Group the average SRs by splits

for (split, cluster), avg_sr in SR_Average_dict.items():
    if split not in SR_split_dict:
        SR_split_dict[split] = []
    SR_split_dict[split].append((cluster, avg_sr))

#=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=

SR_Ranking_dict = {}    # Sort and rank the SRs within each split

for split, sr_list in SR_split_dict.items():
    SR_Ranking_dict[split] = sorted(sr_list, key=lambda x: x[1], reverse=True)

#=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=

print("Ranked average SRs for each split:")

for split, ranked_sr in SR_Ranking_dict.items():
    print('\n\n' + '=*'*50 + '\n' + ' '*40 + f"Split: {split}\n" + '=*'*50 + '\n\n')
    print(f"{'Rank':<10} {'Cluster':<10} {'Average SR':<20}")   # Print the header for each split
    print('-' * 40)
    for rank, (cluster, avg_sr) in enumerate(ranked_sr, start=1):
        print(f"{rank:<10} {cluster:<10} {avg_sr:<20.6f}")


# 
# ---
# ---

# Algorithms for Cluster Selection

# Algorithm #1: **GREEDY** | Top average Sharpe Ratio in Validation Set

# Separate the clusters with positive Vs. negative Sharpe Ratios for $split \in \{tr, test\}$
# \begin{array}{ll}
# \mathcal{G}_{S R^{+}}^{split} &:=\{g \in \mathcal{G} \mid \overline{S R}_g^{split}>0\} 
# \\[1em]
# \mathcal{G}_{S R^{-}}^{split} &:=\{g \in \mathcal{G} \mid \overline{S R}_g^{split}<0\} 
# \end{array} 

# Initialize the sets for clusters
G_SR_plus_train = set()
G_SR_minus_train = set()
G_SR_plus_val = set()
G_SR_minus_val = set()
G_SR_plus_test = set()
G_SR_minus_test = set()

# Iterate through the ranked SRs and classify the clusters
for split, ranked_sr in SR_Ranking_dict.items():
    for cluster, avg_sr in ranked_sr:
        if split == 'Train':
            if avg_sr > 0:
                G_SR_plus_train.add(cluster)
            else:
                G_SR_minus_train.add(cluster)
        elif split == 'Validation':
            if avg_sr > 0:
                G_SR_plus_val.add(cluster)
            else:
                G_SR_minus_val.add(cluster)
        elif split == 'Test':
            if avg_sr > 0:
                G_SR_plus_test.add(cluster)
            else:
                G_SR_minus_test.add(cluster)

# Calculate coincidences for positive and negative SR
coincidence_plus = len(G_SR_plus_train & G_SR_plus_val)
coincidence_minus = len(G_SR_minus_train & G_SR_minus_val)

# coincidence_plus = len(G_SR_plus_train & G_SR_plus_val & G_SR_plus_test)
# coincidence_minus = len(G_SR_minus_train & G_SR_minus_val & G_SR_minus_test)

total_coincidences = coincidence_plus + coincidence_minus

# Print the results
print('=' * 100)
print(' '*30 + "Clusters with Positive Average SR")
print('=' * 100)
print(f"{'Train Splits      :':<25} {G_SR_plus_train}")
print(f"{'Validation Splits :':<25} {G_SR_plus_val}")
print(f"{'Test Splits       :':<25} {G_SR_plus_test}")
print('\n' + f"{'Coincidences (Train, Val):':<25} {coincidence_plus}")

print('\n\n' + '=' * 100)
print(' '*30 + "Clusters with Negative Average SR")
print('=' * 100)
print(f"{'Train Splits      :':<25} {G_SR_minus_train}")
print(f"{'Validation Splits :':<25} {G_SR_minus_val}")
print(f"{'Test Splits       :':<25} {G_SR_minus_test}")
print('\n' + f"{'Coincidences (Train, Val):':<25} {coincidence_minus}")

print('\n\n' + '='*100)
print(f"{'Total Coincidences:':<25} {total_coincidences}/{k_opt}")
print('='*100)


# Obtain the sets of long-traded clusters $\mathcal{G}_{\theta}^+ $ and short traded clusters $\mathcal{G}_{\theta}^-$

# \begin{array}{ll}
# \mathcal{G}_\theta^{+}:=\left\{g \in \mathcal{G} \mid 1 \leq \mathfrak{R}_g^{\text {val }} \leq \theta^{+}\right\}
# \\[1em]
# \mathcal{G}_{\theta}^- := 
# \{ g \in\mathcal{G} \mid 
# %\varkappa_{\ell} \in \G_{SR^-} \wedge
# k^*-\theta^-
# < \mathfrak{R}_g^{val} \leq 
# k^*
# \}
# \end{array}



# Sort clusters in the validation sample by average SR
sorted_positive_val = sorted(G_SR_plus_val, key=lambda cluster: SR_Average_dict[('Validation', cluster)], reverse=True)
sorted_negative_val = sorted(G_SR_minus_val, key=lambda cluster: SR_Average_dict[('Validation', cluster)], reverse=False)

# Select the top θ positions, upper bounded by the cardinality of the G_SR sets
G_θ_plus = sorted_positive_val[:min(θ, len(sorted_positive_val))]
G_θ_minus = sorted_negative_val[:min(θ, len(sorted_negative_val))]

# Function to print lists in a formatted manner
def print_aligned(title, items):
    print(f"{title:<30} : {', '.join(map(str, items))}")

# Print the results
separator = '\n' + '=' * 100 + '\n'
print_aligned("Clusters with Avg SR > 0", sorted_positive_val)
print_aligned("Long-traded Clusters", G_θ_plus)
print(separator)
print_aligned("Clusters with Avg SR < 0", sorted_negative_val)
print_aligned("Short-traded Clusters", G_θ_minus)


# Add a column in $\mathcal B$ with the Trading Rule: [TradingRule]
# $$
# T R_{L, \theta}\langle (i, j), d \rangle 
# :=\left\{\begin{array}{rll}+1 & \text { if } & {\left[(i, j) \in \mathcal{B}_g \wedge g \in \mathcal{G}_\theta^{+}\right] \wedge d \in \mathcal H^i} 
# \\[1em]
# 0 & \text { if } & {\left[(i, j) \in \mathcal{B}_g \wedge g \notin \mathcal{G}_\theta\right] ~~\vee d \notin \mathcal H^i } 
# \\[1em]
# -1 & \text { if } & {\left[(i, j) \in \mathcal{B}_g \wedge g \in \mathcal{G}_\theta^{-}\right] \wedge d \in \mathcal H^i }\end{array}\right.
# $$



B['TR_Greedy'] = 0
B.loc[B['cluster'].isin(G_θ_plus), 'TR_Greedy']   = 1
B.loc[B['cluster'].isin(G_θ_minus), 'TR_Greedy']  = -1

B

# Algorithm #2: **RANK-STABLE** | Minimum Rank Difference between Train and Validation

Split1 = 'Train'
Split2 = 'Validation'

# Calculate the ranks for each cluster in each split
ranks = {split: {cluster: rank for rank, (cluster, _) in enumerate(ranked_sr, start=1)}
         for split, ranked_sr in SR_Ranking_dict.items()}

# Extract the common clusters across Split1 and Split2
common_clusters = set(ranks[Split1]).intersection(ranks[Split2])

# Calculate Spearman rank correlation coefficient for the ranks between Split1 and Split2
split1_ranks = []
split2_ranks = []
for cluster in common_clusters:
    split1_ranks.append(ranks[Split1][cluster])
    split2_ranks.append(ranks[Split2][cluster])

# Compute the Spearman rank correlation
rank_correlation, _ = scipy.stats.spearmanr(split1_ranks, split2_ranks)

# Sort clusters by Split1 ranks to ensure consistency in plotting
sorted_common_clusters = sorted(common_clusters, key=lambda cluster: ranks[Split1][cluster])

#=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=

# Print the rank correlation
print('=*'*50 + '\n' +  f"Spearman Rank Correlation between {Split1} and {Split2} splits: {rank_correlation:.6f}" + '\n' + '=*'*50 + '\n')
print(f"{'Cluster':<10} {Split1 + ' Rank':<15} {Split2 + ' Rank':<15}")
print('-' * 40)

# Print the clusters and their ranks in both splits
for cluster in sorted_common_clusters:
    print(f"{cluster:<10} {ranks[Split1][cluster]:<15} {ranks[Split2][cluster]:<15}")


rank_differences        = {cluster: abs(ranks[Split1][cluster] - ranks[Split2][cluster]) for cluster in common_clusters}        # Calculate the absolute rank differences between Split1 and Split2
sorted_rank_differences = sorted(rank_differences.items(), key=lambda x: x[1])                                                  # Sort clusters by the absolute rank differences
most_stable_clusters    = [cluster for cluster, _ in sorted_rank_differences[:2*θ]]                                           # Select the top θ clusters with the smallest rank differences

print('=*'*50 + f"\nTop {2*θ} most stable clusters between {Split1} and {Split2} splits:" + '\n' + '=*'*50)
for cluster in most_stable_clusters:
    print(f"  ‣ Cluster: {cluster:<3}     {Split1} Rank: {ranks[Split1][cluster]:<3}     {Split2} Rank: {ranks[Split2][cluster]:<3}     Rank Difference: {rank_differences[cluster]:<3}")


# Determine long and short positions based on average Sharpe Ratios
long_clusters = [cluster for cluster in most_stable_clusters if SR_Average_dict[(Split1, cluster)] > 0 and SR_Average_dict[(Split2, cluster)] > 0]
short_clusters = [cluster for cluster in most_stable_clusters if SR_Average_dict[(Split1, cluster)] < 0 and SR_Average_dict[(Split2, cluster)] < 0]

print('Cluster Selection \n' + '-='*50)
print(f" ‣ Clusters selected for Long positions (positive Sharpe Ratios in both splits):    {long_clusters}")
print(f" ‣ Clusters selected for Short positions (negative Sharpe Ratios in both splits):   {short_clusters}")


B['TR_RankStable'] = 0
B.loc[B['cluster'].isin(long_clusters), 'TR_RankStable']   = 1
B.loc[B['cluster'].isin(short_clusters), 'TR_RankStable']  = -1
B

# Look at the correlation between TR_Greedy and TR_RankStable
correlation = B['TR_Greedy'].corr(B['TR_RankStable'])
print(f"Correlation between TR_Greedy and TR_RankStable: {correlation:.6f}")

# 
# ---
# ---

# Table of Clusters & Position of each algorithm on them (LONG/SHORT)

save_this_table = True

#================================================================================================

# Generate LaTeX table with extended columns for Greedy and Stable
latex_table_extended = (
    "\\begin{table}[H]\n\\caption{Mapping of LLM-Shock-Classification to Clusters with Trading Rules}\n\\centering\n\\begin{tabular}{|C{1cm}|l|c|c|}\n\\hline\n"
    "\\rowcolor{gray!10}\n \\multicolumn{2}{|c|}{\\textbf{Cluster}} & \\textbf{Greedy} & "
    "\\textbf{Stable} \\\\ \\hline \\Xhline{2\\arrayrulewidth} \n"
)

# Initialize row counter
row_counter = 0
for shock, cluster in cluster_map.items():
    tuple_str = str(shock).replace("'", "")
    shock_ = r"{" + tuple_str + "}"
    
    # Determine the trading status for Greedy algorithm
    if cluster in G_θ_plus:
        greedy_status = r"\textcolor{darkgreen}{\textsc{long}}"
    elif cluster in G_θ_minus:
        greedy_status = r"\textcolor{darkred}{\textsc{short}}"
    else:
        greedy_status = ""
    
    # Determine the trading status for Stable algorithm
    if cluster in long_clusters:
        stable_status = r"\textcolor{darkgreen}{\textsc{long}}"
    elif cluster in short_clusters:
        stable_status = r"\textcolor{darkred}{\textsc{short}}"
    else:
        stable_status = ""
    
    latex_table_extended += f"{cluster} & {shock_} & {greedy_status} & {stable_status} \\\\ \\hline\n"
    row_counter += 1
    # Add bold \hline every 4 rows
    if row_counter % 4 == 0:
        latex_table_extended += "\\Xhline{2\\arrayrulewidth}\n"

latex_table_extended += (
    "\\end{tabular}"
    "\n\\vspace{0.5cm}"
    "\n\\subcaption*{\\textit{Note: The 'Shock' column reports all the combinations in $\\mathcal{G}_{LLM}$, the 'Cluster' column corresponds to the mapping $\\mathcal{G}_{LLM}\\to\mathbb{N}_0$. The 'Greedy' and 'Stable' columns report the long- and short- traded clusters for the Greedy and Stable algorithms, respectively.}}"
    "\n\\label{tab:LLM_cluster_mapping_extended}\n\\end{table}"
)

# Print LaTeX table
print(latex_table_extended)

if save_this_table:
    with open(os.path.join(path_output, f"{model}_Traded_Clusters.tex"), 'w') as file:
        file.write(latex_table_extended)


# 
# ---
# ---

# Portfolio

# Obtain the trading days associated to each split:

# | Description        | Timeline                        |
# |--------------------|---------------------------------|
# | `timeline`         |  $\tilde{\mathfrak{d}}$         |
# | `timeline_train`   | $\tilde{\mathfrak{d}}^{train}$  |
# | `timeline_val`     | $\tilde{\mathfrak{d}}^{val}$    |
# | `timeline_test`    | $\tilde{\mathfrak{d}}^{test}$   |


# Construct the Portfolio
# $$\mathcal{P}:=\left\{\langle(i, j), d\rangle \mid(i, j) \in \mathcal{B} \wedge d \in \tilde{\mathfrak{d}} \wedge T R_{L, \theta}\langle(i, j), d\rangle \neq 0\right\}$$

# The set of open positions on a particular day $d \in \tilde{\mathfrak{d}}$ is defined as
# $$
# \mathcal{P}_d:=\left\{(i, j) \in \mathcal{B} \mid d \in \tilde{\mathfrak{d}} \wedge T R_{L, \theta}\langle(i, j), d\rangle \neq 0\right\}
# $$
# where $\left|\mathcal{P}_d\right|=\sum_{(i, j) \in \mathcal{B}}\left|T R_{L, \theta}\langle(i, j), d\rangle\right|$


# The return of the portfolio on trading date $d$, denoted as $r_{d}^{\mathcal{P}}$, is the sum of the returns of the individual positions weighted by their trading signals:

# $$r_d^{\mathcal{P}}=\frac{1}{\left|\mathcal{P}_d\right|} \sum_{(i, j), \in \mathcal{P}_d} T R_{L, \theta}\langle(i, j), d\rangle \cdot A R_d^{(i, j)}$$

# Recall that positions are only held for $L$ trading days: 
# $d \in \mathcal H^i$
# where $\mathcal{H}^i:=\left\{d \in \tilde{\mathfrak{d}} \mid \tilde{d}_0^i \leq d \leq \mathbb{D}_{\tilde{\mathfrak{d}}}\left(\mathbb{I}_{\tilde{\mathfrak{d}}}\left(\tilde{d}_0^i\right)+L\right)\right\}$

def Initialize_Portfolio(B, 𝖉):
# Side function to initialize the portfolio DataFrames in the main function `calculate_portfolio_returns``
    """
    Initialize the portfolio DataFrames for train, validation, and test splits.

    Parameters:
    D_train (pd.DataFrame): DataFrame for training data.
    D_val (pd.DataFrame): DataFrame for validation data.
    D_test (pd.DataFrame): DataFrame for test data.
    𝖉 (list): List of trading days.

    Returns:
    dict: Dictionary containing initialized portfolio DataFrames and trading day timelines for each split and the overall sample.
    """
    B_train = B[B['split'] == 'Train']       
    B_val   = B[B['split'] == 'Validation']
    B_test  = B[B['split'] == 'Test']

    splits = {
        'train': B_train,
        'val':   B_val,
        'test':  B_test
    }

    indices = {}
    
    for split_name, split_data in splits.items():
        first_day = split_data['date_affect'].min()
        last_day = split_data['date_affect'].max()

        indices[f'first_day_{split_name}_index'] = 𝖉.index(first_day)
        indices[f'last_day_{split_name}_index'] = 𝖉.index(last_day)

    # Extract individual indices
    first_day_train_index = indices['first_day_train_index']
    last_day_train_index = indices['last_day_train_index']
    first_day_val_index = indices['first_day_val_index']
    last_day_val_index = indices['last_day_val_index']
    first_day_test_index = indices['first_day_test_index']
    last_day_test_index = indices['last_day_test_index']

    # Create trading day timelines
    𝖉_all = 𝖉[first_day_train_index:last_day_test_index + 1]
    𝖉_train = 𝖉[first_day_train_index:last_day_train_index + 1]
    𝖉_val = 𝖉[last_day_train_index:last_day_val_index + 1]
    𝖉_test = 𝖉[last_day_val_index:last_day_test_index + 1]
    
    # Initialize DataFrames to save the portfolio returns
    r_P_all = pd.DataFrame(0, index=𝖉_all, columns=['returns'])
    r_P_train = pd.DataFrame(0, index=𝖉_train, columns=['returns'])
    r_P_val = pd.DataFrame(0, index=𝖉_val, columns=['returns'])
    r_P_test = pd.DataFrame(0, index=𝖉_test, columns=['returns'])

    r_P_dict = {
        'All': r_P_all,
        'Train': r_P_train,
        'Validation': r_P_val,
        'Test': r_P_test
    }

    𝖉_dict = {
        'All': 𝖉_all,
        'Train': 𝖉_train,
        'Validation': 𝖉_val,
        'Test': 𝖉_test
    }

    return r_P_dict, 𝖉_dict

#=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

def print_header(d̃, index_d̃):
    print('\n\n' + '=*' * 50 + '\n' + f'{d̃ = } | {index_d̃ = }' + '\n' + '=*' * 50)

#=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

def print_portfolio_info(idx, ticker, split, cluster, date_affect, idx_diff, index_d̃, index_date_affect, trading_rule, AR, TS_data):
    print(f"""
  ► (i,j) index in 𝓑    = {idx}
  ► Ticker              = {ticker}
  ► Split               = {split}
  ► Cluster             = {cluster}
  ► d̃_0                 = {date_affect}
  ► d̃ - d̃_0             = {idx_diff} = [index of d̃ in 𝖉_all ({index_d̃})] - [index of d̃_0 in 𝖉_all ({index_date_affect})]""")
    if TS_data is not None and isinstance(TS_data, pd.DataFrame):
        print(f"""  -------------------------------------------------
  ► Trading Rule        = {trading_rule}
  ► Abnormal Return     = {AR}
  -------------------------------------------------
  ► Position Returns    = {AR * trading_rule}
                      """)
    else:
        print(f"""
  ► Position Returns    = [✕] No Data available for this ticker'
                      """)


def calculate_portfolio_returns(B, 𝖉, L, TS_dict, TradingRule='TR_RankStable', trading_cost_bps=15, verbose=True):
    # Initialize portfolio DataFrames and trading day timelines
    r_P_dict, 𝖉_dict = Initialize_Portfolio(B, 𝖉)
    r_P_all = r_P_dict['All']
    𝖉_all = 𝖉_dict['All']
    
    # Create DataFrames for trading signal evolution and turnover
    trading_signal_evolution = {split: pd.DataFrame(index=𝖉_dict[split], 
                                                  columns=['total_trading_signal']) 
                              for split in ['All', 'Train', 'Validation', 'Test']}
    
    # Initialize DataFrames for tracking positions and turnover
    position_tracking = pd.DataFrame(index=𝖉_all, columns=['total_trading_signal'])
    turnover_tracking = pd.DataFrame(index=𝖉_all, columns=['turnover'])
    previous_positions = {}
    
    # Initialize both gross and net returns
    for split in r_P_dict:
        if isinstance(r_P_dict[split], pd.DataFrame):  # Check if it's a DataFrame
            r_P_dict[split]['gross_returns'] = pd.Series(index=r_P_dict[split].index, dtype=float)
            r_P_dict[split]['net_returns'] = pd.Series(index=r_P_dict[split].index, dtype=float)
            if 'returns' in r_P_dict[split].columns:
                del r_P_dict[split]['returns']
    
    for d̃ in 𝖉_all:
        index_d̃ = 𝖉_all.index(d̃)
        print_header(d̃, index_d̃) if verbose else None
        
        # Define Portfolio subset
        L_trading_days_before_d̃ = 𝖉[𝖉.index(d̃) - L]
        Portfolio = B[(B['date_affect'] >= L_trading_days_before_d̃) & 
                     (B['date_affect'] <= d̃) & 
                     (B[TradingRule] != 0)]
        
        total_trading_signal = 0
        total_weighted_return = 0
        current_positions = {}
        
        if len(Portfolio) > 0:  # Changed from Portfolio.empty
            for idx, row in Portfolio.iterrows():
                ticker = row['tickers']
                split = row['split']
                date_affect = row['date_affect']
                cluster = row['cluster']
                trading_rule = row[TradingRule]
                
                current_positions[ticker] = trading_rule
                
                index_date_affect = 𝖉_all.index(date_affect)
                idx_diff = index_d̃ - index_date_affect
                
                TS_data = TS_dict.get(idx)
                if TS_data is not None and isinstance(TS_data, pd.DataFrame):
                    AR = TS_data['AR'].iloc[idx_diff]
                    total_weighted_return += AR * trading_rule
                    total_trading_signal += abs(trading_rule)
                    
                    if verbose:
                        print_portfolio_info(idx, ticker, split, cluster, date_affect,
                                          idx_diff, index_d̃, index_date_affect,
                                          trading_rule, AR, TS_data)
        
        # Calculate turnover
        daily_turnover = 0
        if previous_positions:
            position_changes = 0
            total_position_size = sum(abs(pos) for pos in current_positions.values())
            
            all_tickers = set(current_positions.keys()) | set(previous_positions.keys())
            for ticker in all_tickers:
                curr_pos = current_positions.get(ticker, 0)
                prev_pos = previous_positions.get(ticker, 0)
                position_changes += abs(curr_pos - prev_pos)
            
            if total_position_size > 0:
                daily_turnover = position_changes / total_position_size
        
        # Store turnover
        turnover_tracking.loc[d̃, 'turnover'] = daily_turnover
        previous_positions = current_positions.copy()
        
        # Store the trading signal evolution
        trading_signal_evolution['All'].loc[d̃, 'total_trading_signal'] = total_trading_signal
        for split in ['Train', 'Validation', 'Test']:
            if d̃ in 𝖉_dict[split]:
                trading_signal_evolution[split].loc[d̃, 'total_trading_signal'] = total_trading_signal
        
        # Calculate returns
        gross_return = total_weighted_return / total_trading_signal if total_trading_signal != 0 else 0
        trading_costs = daily_turnover * (trading_cost_bps / 10000)
        net_return = gross_return - trading_costs
        
        if verbose:
            print('_' * 100 + '\n' + 'Returns and costs for this day:')
            print(f' ➤ Gross return: {gross_return:.4f}')
            print(f' ➤ Trading costs: {trading_costs:.4f}')
            print(f' ➤ Net return: {net_return:.4f}')
            print(f' ➤ Daily turnover: {daily_turnover:.4f}')
        
        # Store returns in DataFrames
        r_P_dict['All'].loc[d̃, 'gross_returns'] = gross_return
        r_P_dict['All'].loc[d̃, 'net_returns'] = net_return
        
        for split in ['Train', 'Validation', 'Test']:
            if d̃ in 𝖉_dict[split]:
                r_P_dict[split].loc[d̃, 'gross_returns'] = gross_return
                r_P_dict[split].loc[d̃, 'net_returns'] = net_return
    
    # Calculate average turnover by split
    split_dates = {
        'All': 𝖉_all,
        'Train': list(𝖉_dict['Train']),
        'Validation': list(𝖉_dict['Validation']),
        'Test': list(𝖉_dict['Test'])
    }
    
    turnover_stats = {}
    for split, dates in split_dates.items():
        split_turnover = turnover_tracking.loc[dates, 'turnover']
        turnover_stats[split] = split_turnover.mean()
    
    # Add metrics to the return dictionary
    r_P_dict['trading_signal_evolution'] = trading_signal_evolution
    r_P_dict['turnover'] = turnover_tracking
    r_P_dict['turnover_stats'] = turnover_stats
    
    return r_P_dict

# Main execution code remains the same as before
TradingRule_dict = {
    'Greedy': 'TR_Greedy',
    'Stable': 'TR_RankStable'
}

# Initialize dictionaries
r_P_all, r_P_train, r_P_val, r_P_test = {}, {}, {}, {}
trading_signal_evolution_all = {}
turnover_stats_all = {}

# Calculate portfolio returns for each algorithm
for key, TR in TradingRule_dict.items():
    r_P_dict = calculate_portfolio_returns(B, 𝖉, L, TS_dict, TradingRule=TR, verbose=True)
    
    # Store returns (both gross and net)
    r_P_all[key] = {
        'gross': r_P_dict['All']['gross_returns'],
        'net': r_P_dict['All']['net_returns']
    }
    r_P_train[key] = {
        'gross': r_P_dict['Train']['gross_returns'],
        'net': r_P_dict['Train']['net_returns']
    }
    r_P_val[key] = {
        'gross': r_P_dict['Validation']['gross_returns'],
        'net': r_P_dict['Validation']['net_returns']
    }
    r_P_test[key] = {
        'gross': r_P_dict['Test']['gross_returns'],
        'net': r_P_dict['Test']['net_returns']
    }
    
    trading_signal_evolution_all[key] = r_P_dict['trading_signal_evolution']
    turnover_stats_all[key] = r_P_dict['turnover_stats']

# Print turnover statistics
print("\nPortfolio Turnover Statistics:")
print("-" * 50)
for algo in ['Greedy', 'Stable']:
    print(f"\n{algo} Algorithm:")
    for split, turnover in turnover_stats_all[algo].items():
        print(f"{split} Split Average Turnover: {turnover:.4f}")

# Convert into dataframes with both gross and net returns
r_P_all = pd.DataFrame({
    (key, ret_type): data[ret_type] 
    for key, data in r_P_all.items() 
    for ret_type in ['gross', 'net']
})
r_P_train = pd.DataFrame({
    (key, ret_type): data[ret_type] 
    for key, data in r_P_train.items() 
    for ret_type in ['gross', 'net']
})
r_P_val = pd.DataFrame({
    (key, ret_type): data[ret_type] 
    for key, data in r_P_val.items() 
    for ret_type in ['gross', 'net']
})
r_P_test = pd.DataFrame({
    (key, ret_type): data[ret_type] 
    for key, data in r_P_test.items() 
    for ret_type in ['gross', 'net']
})

# Final dictionary
r_P_dict = {
    'All': r_P_all,
    'Train': r_P_train,
    'Validation': r_P_val,
    'Test': r_P_test,
    'trading_signal_evolution': trading_signal_evolution_all,
    'turnover_stats': turnover_stats_all
}

# Trading Intensity

label                   = "LLM_trading_intensity"
# path_output             = ""

def create_trading_intensity_table(r_P_dict, trading_signal_evolution_all, turnover_stats_all):
    """
    Create a comprehensive table of trading intensity metrics.
    Returns both DataFrame and LaTeX formats.
    """
    # Create list to store data
    table_data = []
    
    for split in ['All', 'Train', 'Validation', 'Test']:
        for algo in ['Greedy', 'Stable']:
            # Get position data
            positions = trading_signal_evolution_all[algo][split]['total_trading_signal']
            
            # Calculate trading costs
            trading_costs = (r_P_dict[split][f'{algo}', 'gross'] - 
                           r_P_dict[split][f'{algo}', 'net']).mean() * 100  # as percentage
            
            # Calculate percentage of days with trading activity
            active_days = (positions > 0).mean() * 100  # as percentage
            
            metrics = {
                'Split': split,
                'Algorithm': algo,
                # Position metrics
                'Avg. Positions': positions.mean(),
                'Position Std.': positions.std(),
                'Max Positions': positions.max(),
                'Min Positions': positions.min(),
                # Trading intensity metrics
                'Turnover': turnover_stats_all[algo][split] * 100,  # as percentage
                'Changes/Position': turnover_stats_all[algo][split] / positions.mean() * 100,  # as percentage
                # Cost metrics
                'Avg. Costs (%)': trading_costs,
                # Activity metrics
                'Active Days (%)': active_days
            }
            
            table_data.append(metrics)
    
    # Create DataFrame
    metrics_df = pd.DataFrame(table_data)
    
# The LaTeX table generation with detailed caption
    latex_table = "\\inserthere{tab:" + label + "}\n\n"
    latex_table += "\\begin{table}[htbp] \n"
    latex_table += "\caption{Trading Intensity Analysis: " + model + "} \n"
    latex_table += "\\centering \n"
    latex_table += "\\label{tab:" + label + "}"
    latex_table += """
{\small
\\begin{tabular}{lcccccccccc}
\\toprule
Split & Algorithm & \\multicolumn{4}{c}{\\# Open Positions} & \\multicolumn{2}{c}{Trading Activity (\\%)} & \\multicolumn{2}{c}{Trading Costs (\\%)} \\\\
\\cmidrule(lr{0.6em}){3-6} \\cmidrule(lr{0.6em}){7-8} \\cmidrule(lr{0.6em}){9-10}
& & Avg. & Std. & Max & Min & Turnover & Changes/Pos. & Cost & Active \\\\
\\midrule
"""
    # Add data rows with multirow for splits
    for split in ['All', 'Train', 'Validation', 'Test']:
        # Filter data for this split
        split_data = metrics_df[metrics_df['Split'] == split]
        
        # Add Greedy row with multirow split
        greedy_row = split_data[split_data['Algorithm'] == 'Greedy'].iloc[0]
        latex_table += f"\\multirow{{2}}{{*}}{{{split}}} & \\textit{{Greedy}} & "
        latex_table += f"{greedy_row['Avg. Positions']:.1f} & {greedy_row['Position Std.']:.2f} & "
        latex_table += f"{greedy_row['Max Positions']:.0f} & {greedy_row['Min Positions']:.0f} & "
        latex_table += f"{greedy_row['Turnover']:.2f} & {greedy_row['Changes/Position']:.3f} & "
        latex_table += f"{greedy_row['Avg. Costs (%)']:.4f} & {greedy_row['Active Days (%)']:.1f} \\\\\n"
        
        # Add Stable row
        stable_row = split_data[split_data['Algorithm'] == 'Stable'].iloc[0]
        latex_table += f" & \\textit{{Stable}} & "
        latex_table += f"{stable_row['Avg. Positions']:.2f} & {stable_row['Position Std.']:.2f} & "
        latex_table += f"{stable_row['Max Positions']:.0f} & {stable_row['Min Positions']:.0f} & "
        latex_table += f"{stable_row['Turnover']:.2f} & {stable_row['Changes/Position']:.3f} & "
        latex_table += f"{stable_row['Avg. Costs (%)']:.4f} & {stable_row['Active Days (%)']:.1f} \\\\\n"
        
        # Add midrule except after last split
        if split != 'Test':
            latex_table += "\\midrule\n"
    
    # Close the table with bottomrule
    latex_table += """\\bottomrule
\\end{tabular}
}

\\vspace{0.5cm}
\\begin{minipage}{\\textwidth}
\\setlength{\\parindent}{0pt}
\\scriptsize\\textit{Note}: 
This table presents trading intensity metrics for both Greedy and Stable algorithms across different data splits. 
The metrics are computed at a daily frequency. The `\\# Open Positions' columns report position-related statistics: 
`Avg.' shows the mean number of concurrent open positions per day, `Std.' represents their standard deviation, while 
`Max' and `Min' indicate the maximum and minimum number of positions held simultaneously. Under `Trading Activity (\\%)', 
`Turnover' is calculated as the sum of absolute changes in position sizes divided by the total portfolio size, expressed 
as a percentage; formally, $Turnover_t = 100 \\times (\\sum_i |w_{i,t} - w_{i,t-1}|)/(\\sum_i |w_{i,t}|)$, where $w_{i,t}$ 
represents the position size in asset $i$ at time $t$. `Changes/Pos.' represents the average number of modifications per 
position per day, computed as the daily turnover divided by the average number of positions, providing insight into how 
actively individual positions are managed. The `Trading Costs (\\%)' section reports `Cost' as the average daily implementation 
shortfall (computed as the difference between gross and net returns) expressed in percentage terms, while `Active' shows 
the percentage of trading days with at least one open position. All metrics are first computed daily and then averaged 
over their respective periods, except for Max and Min positions which represent the absolute extremes over each period.
\\end{minipage}
\\end{table}"""

##########################################################################################
    return {
        'dataframe': metrics_df,
        'latex': latex_table
    }

# Generate the table
results = create_trading_intensity_table(r_P_dict, trading_signal_evolution_all, turnover_stats_all)

# Print DataFrame version for checking
# print("\nTrading Intensity Metrics DataFrame:")
# print("=" * 80)
# print(results['dataframe'].to_string(index=False))
# print("\nLaTeX Table has been generated and stored in results['latex']")

latex_table = results['latex']
print(latex_table)


# save the latex table to path_output under the name f"tab_{model}_Trading_Intensity.tex"
with open(f"{path_output}/tab_{model}_Trading_Intensity.tex", "w") as text_file:
    text_file.write(latex_table)


# Open Positions per Day

TICK_FONTSIZE = 24
LABEL_FONTSIZE = 24
LEGEND_FONTSIZE = 20
# path_output = "/Users/jesusvillotamiranda/Library/CloudStorage/OneDrive-UniversidaddeLaRioja/GitHub/Repository/TeX_Repo/__JBF_Submission__"

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import numpy as np

# Custom date formatter
class CustomDateFormatter(mdates.DateFormatter):
    def __init__(self, fmt="%b-%Y", *args, **kwargs):
        super().__init__(fmt, *args, **kwargs)
        self.fmt = fmt
        self.last_year = None
        
    def __call__(self, x, pos=0):
        dt = mdates.num2date(x)
        if self.last_year != dt.year:
            self.last_year = dt.year
            return dt.strftime("%b\n%Y")
        else:
            return dt.strftime("%b")  # Months in TEXTUAL format

# Initialize the plot
fig, ax = plt.subplots(figsize=(14, 10))

# Set the background color to grey
ax.set_facecolor('#f5f5f5')

# Define colors for the trading rules
colors = {'Greedy': 'blue', 'Stable': 'green'}

# Plot number of open positions for each algorithm
for algo in ['Greedy', 'Stable']:
    data = trading_signal_evolution_all[algo]['All']
    ax.plot(data.index, data['total_trading_signal'],
            label=algo,
            color=colors[algo],
            linestyle='-')

# Set y-axis limits from 0 to 110
ax.set_ylim(0, 110)

# Add vertical lines for each split
for split in ['Train', 'Validation', 'Test']:
    split_start = trading_signal_evolution_all['Greedy'][split].index[0]
    ax.axvline(x=split_start, color='grey', linestyle='--', linewidth=1)
    # Adjust text position to work with new y-axis limits
    ax.text(split_start, 111, f'\\textit{{{split}}}',
            horizontalalignment='center', verticalalignment='bottom',
            fontsize=TICK_FONTSIZE, fontstyle='italic')
    if split == 'Test':
        end_split = split_start
        # Shade area to the right of the "Test" split
        ax.axvspan(end_split, trading_signal_evolution_all['Greedy']['Test'].index[-1],
                  facecolor='peachpuff', alpha=0.3)

# Customize grid
ax.grid(True, alpha=0.3, linestyle=':')

# Create proxy artists for the legend
legend_elements = [Line2D([0], [0], color='blue', lw=2, label='Greedy'),
                  Line2D([0], [0], color='green', lw=2, label='Stable')]

# Adjust the bottom margin
plt.subplots_adjust(bottom=0.3)

# Add labels and title
ax.set_xlabel('Time (\\textit{trading days})', fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
ax.set_ylabel('\\# Open Positions', fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set tick parameters
ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
ax.set_xlim(right=trading_signal_evolution_all['Greedy']['Test'].index[-1])

# Set x-axis formatter
ax.xaxis.set_major_formatter(CustomDateFormatter())

# Set title if not saving
if not save_this_plot:
    ax.set_title(f'Evolution of Open Positions $~\mid~ (L={L},~\\theta = \\lfloor {prop_of_k}k \\rfloor)$ ',
                fontsize=TITLE_FONTSIZE, pad=50)

# Add legend
ax.legend(handles=legend_elements, fontsize=LEGEND_FONTSIZE)

# Save the plot if requested
if save_this_plot:
    plt.savefig(os.path.join(path_output, f'fig_{model}_Open_Positions.pdf'),
                bbox_inches='tight')

# Show the plot
plt.show()

# Portfolio Metrics

# Portfolio returns:
# $$\mu^{\mathcal{P}}=\frac{1}{|\tilde{\mathfrak{d}}|} \sum_{d \in \tilde{\mathfrak{d}}} \ln \left(1+r_{d}^{\mathcal{P}}\right)$$
# The standard deviation:
# $$ \sigma^{\mathcal{P}}=\sqrt{\frac{1}{|\tilde{\mathfrak{d}}|-1} \sum_{d \in \tilde{\mathfrak{d}}}\left[\ln \left(1+r_{d}^{\mathcal{P}}\right)-\mu^{\mathcal{P}}\right]^2} $$
# The annualized Sharpe Ratio:
# $$ SR^{\mathcal{P}}=\frac{\mu^{\mathcal{P}}}{\sigma^{\mathcal{P}}} \sqrt{252} $$

def portfolio_statistics(r_P):
    from scipy import stats
    import numpy as np
    # Cumulative return series
    cum_r_P_series = (1 + r_P).cumprod()

    # Final cumulative return
    # cum_r_P_final = cum_r_P_series.iloc[-1]
    cum_r_P_final = (1 + r_P).prod()

    # Compute log returns
    log_returns = np.log(1 + r_P)

    # Annualized metrics
    ann_factor = 252
    μ_P = (1 + r_P).prod() ** (ann_factor / len(r_P)) - 1
    σ_P = np.std(log_returns, ddof=1) * np.sqrt(ann_factor)
    
    # Sharpe Ratio
    SR_P = (np.mean(log_returns) / np.std(log_returns, ddof=1)) * np.sqrt(ann_factor)
    # SR_P = (np.mean(r_P) / np.std(r_P, ddof=1)) * np.sqrt(ann_factor) # Very similar to the above line
    
    # Sortino Ratio
    downside_returns = r_P[r_P < 0]
    downside_deviation = np.sqrt(np.mean(downside_returns**2) * ann_factor)
    Sortino_Ratio = μ_P / downside_deviation if downside_deviation != 0 else np.nan

    # Maximum Drawdown (MDD)
    running_max = cum_r_P_series.cummax()
    drawdowns = (cum_r_P_series - running_max) / running_max
    MDD = drawdowns.min() # Maximum Drawdown as a positive value

    # Calmar Ratio
    N_days = len(r_P)
    annualized_return = (cum_r_P_final) ** (252 / N_days) - 1
    Calmar_Ratio = annualized_return / abs(MDD) if MDD != 0 else np.nan

    # Skewness and Kurtosis of log returns
    # skewness = stats.skew(log_returns)
    # kurtosis = stats.kurtosis(log_returns, fisher=True) # Excess Kurtosis = Regular Kurtosis - 3 (Fisher's definition)
    ann_returns = r_P * np.sqrt(ann_factor)
    skewness = stats.skew(ann_returns)
    kurtosis = stats.kurtosis(ann_returns, fisher=True) # Excess Kurtosis = Regular Kurtosis - 3 (Fisher's definition)

    # Value at Risk (VaR) and Conditional Value at Risk (CVaR) - Assuming 95% Confidence Level => alpha = 0.05
    alpha = 0.05
    
    VaR_95 = np.percentile(r_P, 100 * alpha)
    tail_returns = r_P[r_P <= VaR_95]
    CVaR_95 = tail_returns.mean() if len(tail_returns) > 0 else np.nan
    
    VaR_95 = VaR_95 * np.sqrt(ann_factor) # Annualize the VaR
    CVaR_95 = CVaR_95 * np.sqrt(ann_factor) # Annualize the CVaR

    return {
        'cum_r_P_final': cum_r_P_final,
        'μ_P': μ_P,
        'σ_P': σ_P,
        'SR_P': SR_P,
        'Sortino_Ratio': Sortino_Ratio,  
        'MDD': MDD,
        'Calmar_Ratio': Calmar_Ratio,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'VaR_95': VaR_95,
        'CVaR_95': CVaR_95
    }
    

P_statistics = {
    'All': {}, 'Train': {}, 'Validation': {}, 'Test': {}
}

# Loop to apply portfolio_statistics function to each dataframe in r_P_dict
for split, r_P in r_P_dict.items():
    if split not in ['trading_signal_evolution', 'turnover_stats']:  # Skip non-return entries
        for TR_key in TradingRule_dict.keys():
            P_statistics[split][TR_key] = {
                'gross': portfolio_statistics(r_P[(TR_key, 'gross')]),
                'net': portfolio_statistics(r_P[(TR_key, 'net')])
            }

# Calculate cumulative returns for each split, trading rule, and return type
cumulative_returns = {
    'Train': {
        'gross': (1 + pd.DataFrame({TR: r_P_train[(TR, 'gross')] for TR in TradingRule_dict.keys()})).cumprod(),
        'net': (1 + pd.DataFrame({TR: r_P_train[(TR, 'net')] for TR in TradingRule_dict.keys()})).cumprod()
    },
    'Validation': {
        'gross': (1 + pd.DataFrame({TR: r_P_val[(TR, 'gross')] for TR in TradingRule_dict.keys()})).cumprod(),
        'net': (1 + pd.DataFrame({TR: r_P_val[(TR, 'net')] for TR in TradingRule_dict.keys()})).cumprod()
    },
    'Test': {
        'gross': (1 + pd.DataFrame({TR: r_P_test[(TR, 'gross')] for TR in TradingRule_dict.keys()})).cumprod(),
        'net': (1 + pd.DataFrame({TR: r_P_test[(TR, 'net')] for TR in TradingRule_dict.keys()})).cumprod()
    }
}

caption                 = r"$\mathcal{P}_{\text{LLM}}$"
label                   = "LLM_portfolio_statistics"
subcaption_specific1    = r"Portfolio statistics of the trading strategy applied to the LLM clusters."
subcaption_specific2    = r"The holding period of the beta-neutral positions is set to $L$ = 4 trading days and the number of traded clusters is, $\theta = 0.5k=10$ as there are $k^*=20$ LLM clusters of article embeddings. The selection criteria for these hyperparameters ($L,\theta$) is based on maximizing the Sharpe Ratios of the train and validation samples."
# path_output             = "/Users/jesusvillotamiranda/Library/CloudStorage/OneDrive-UniversidaddeLaRioja/GitHub/Repository/TeX_Repo/__JBF_Submission__"

def generate_portfolio_tables(P_statistics, label, caption, subcaption_specific1, subcaption_specific2, path_output, save_tables=True):
    # Generate tables for both gross and net returns
    for return_type in ['gross', 'net']:
        table = "\\inserthere{tab:" + f"{label}_{return_type}" + "}\n"
        table += '\\begin{table}[H] \n'
        table += "\caption{Statistics of " + caption + " across data splits | " + return_type.capitalize() + " Returns} \n"
        table += r'''
\centering
\renewcommand{\arraystretch}{1.1} % Increased line spacing
# % Define new column types for better spacing  # Jupyter magic command
\newcolumntype{P}[1]{>{\centering\arraybackslash}p{#1}}
{\footnotesize
\begin{tabular}{
 P{1.28cm} % Split
 P{0.9cm} % Algorithm
 P{0.9cm} % Cum. Return
 P{0.9cm} % Avg. Return
 P{0.9cm} % St. Deviation
 P{0.9cm} % Sharpe Ratio
 P{0.9cm} % Sortino Ratio
 P{0.9cm} % Max. Drawdown
 P{1cm} % Calmar Ratio
 P{0.9cm} % Skewness
 P{0.9cm} % Kurtosis
 P{0.9cm} % VaR
 P{0.9cm} % CVaR
}
\Xhline{2\arrayrulewidth}
\textbf{Split} & \textbf{Algo.} & \textbf{Cum. Ret.} & \textbf{Avg. Ret.} & \textbf{St. Dev.} & \textbf{Sharpe Ratio} & \textbf{Sortino Ratio} & \textbf{Max. DD} & \textbf{Calmar Ratio} & \textbf{Skew.} & \textbf{Exc. Kurt.} & \textbf{VaR 95\%} & \textbf{CVaR 95\%} \\
\Xhline{2\arrayrulewidth}
'''
        for split_name, split_dict in P_statistics.items():
            if split_name not in ['trading_signal_evolution', 'turnover_stats']:  # Skip non-return entries
                table += f"\\multirow{{2}}{{*}}{{{split_name}}}"  # Split row with multirow
                for TR_key, TR_stats in split_dict.items():
                    stats = TR_stats[return_type]
                    cum_r_P_final, μ_P, σ_P, SR_P, Sortino_Ratio, MDD, Calmar_Ratio, skewness, kurtosis, VaR_95, CVaR_95 = stats.values()
                    table += f" & \\textit{{{TR_key}}} & {cum_r_P_final:.3f} & {μ_P*100:.1f} & {σ_P*100:.1f} & {SR_P:.1f} & {Sortino_Ratio:.1f} & {MDD*100:.1f} & {Calmar_Ratio:.1f} & {skewness:.2f} & {kurtosis:.2f} & {VaR_95*100:.1f} & {CVaR_95*100:.1f} \\\\ "
                table += r" \hline "

        table += "\n" + r'\end{tabular}' + '\n }'
        table += f"\n \\label{{tab:{label}_{return_type}}}"
        
        # Add notes section
        table += r'''
        
\vspace{0.5cm}
\begin{minipage}{\textwidth}
\setlength{\parindent}{0pt}
{\footnotesize\textit{Note:
'''
        table += subcaption_specific1
        table += r'''
The statistics provided include performance metrics (Cumulative Return, Average Return (\%)), risk measures (Standard Deviation (\%), Maximum Drawdown (\%), Value at Risk (\%), Conditional Value at Risk (\%)), risk-adjusted performance ratios (Sharpe Ratio, Sortino Ratio, Calmar Ratio), and return distribution characteristics (Skewness, Excess Kurtosis). These statistics are provided for both cluster-selection algorithms: Greedy and Stable.
Except for the Cumulative Return, all returns are annualized. The Sharpe Ratio is computed using the daily returns, assuming 252 trading days in a year. The Sortino Ratio is calculated using the daily downside returns. The Maximum Drawdown is the maximum loss from a peak to a trough. The Calmar Ratio is the ratio of the annualized return to the maximum drawdown. Skewness measures the asymmetry of the return distribution, while Kurtosis quantifies the tails' thickness. The Value at Risk (VaR) and Conditional Value at Risk (CVaR) are calculated at a 95\% confidence level.
The Greedy algorithm longs (shorts) clusters that maximize (minimize) the cluster-average-$SR$ in the validation sample subject to a positivity (negativity) constraint, while the Stable algorithm longs (shorts) clusters that minimize the rank difference between the training and validation rankings of the cluster-average-$SR$'s subject to a positivity (negativity) constraint, which is now imposed on both sample splits. In both algorithms, the cardinality of each leg is upper-bounded by a hyperparameter $\theta$.
'''
        table += subcaption_specific2
        table += r'''
}}
\end{minipage}
\end{table}
'''
        # print(f"\nTable for {return_type} returns:")
        print(table)
        
        if save_tables:
            with open(path_output + f'/tab_{model}_Portfolio_Statistics_{return_type.upper()}.tex', 'w') as file:
                file.write(table)

# Usage example:
generate_portfolio_tables(
    P_statistics=P_statistics,
    label=label,
    caption=caption,
    subcaption_specific1=subcaption_specific1,
    subcaption_specific2=subcaption_specific2,
    path_output=path_output,
    save_tables=True
)

save_this_plot = True

#===============================================================================================================

# Custom date formatter
class CustomDateFormatter(mdates.DateFormatter):
    def __init__(self, fmt="%b-%Y", *args, **kwargs):
        super().__init__(fmt, *args, **kwargs)
        self.fmt = fmt
        self.last_year = None

    def __call__(self, x, pos=0):
        dt = mdates.num2date(x)
        if self.last_year != dt.year:
            self.last_year = dt.year
            return dt.strftime("%b\n%Y")
        else:
            return dt.strftime("%b")      # Uncomment for: Months in TEXTUAL format
            # return dt.strftime("%m")    # Uncomment for: Months in NUMERIC format
        
#===============================================================================================================

# Initialize the plot
fig, ax = plt.subplots(figsize=(14, 10))  # (14,8)

# Set the background color to grey
ax.set_facecolor('#f5f5f5')

# Define colors for the trading rules
colors = {'Greedy': 'blue', 'Stable': 'green'}

# Plot cumulative returns for each split and each Trading Rule
for split, returns in cumulative_returns.items():
    for rule in TradingRule_dict.keys():
        ax.plot(returns[rule].index, returns[rule].values, 
                label=rule, 
                color=colors[rule], 
                linestyle='-')

# Add vertical lines for each split and shade area to the right of the "Train" split
for split in ['Train', 'Validation', 'Test']:
    split_start = cumulative_returns[split].index[0]
    ax.axvline(x=split_start, color='grey', linestyle='--', linewidth=1)
    ax.text(split_start, ax.get_ylim()[1] * 1.005, f'\\textit{{{split}}}', 
            horizontalalignment='center', verticalalignment='bottom', fontsize=TICK_FONTSIZE, fontstyle='italic')
    
    if split == 'Test':
        end_split = split_start  # Save the end date of the train split for shading

# Shade area to the right of the "Test" split
ax.axvspan(end_split, cumulative_returns['Test'].index[-1], facecolor='peachpuff', alpha=0.3) ## Colors: ["lightcoral", "lightblue", "lightgreen", "lightyellow", "lightgrey", "lightpink", "lightcyan", "lightgoldenrodyellow", "lavender", "mistyrose", "peachpuff"]


# Add horizontal line at 1
ax.axhline(y=1, color='grey', linestyle='--', linewidth=1)

# Customize grid
ax.grid(True, alpha=0.3, linestyle=':')  # Set grid transparency and linestyle to dotted

# Create proxy artists for the legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='blue', lw=2, label='Greedy'),
                   Line2D([0], [0], color='green', lw=2, label='Stable')]

# Adjust the bottom margin to make room for the labels
plt.subplots_adjust(bottom=0.3)

# Add labels and title
ax.set_xlabel('Time (\\textit{trading days})', fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
ax.set_ylabel('Cumulative Returns', fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set tick parameters
ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
ax.set_xlim(right=cumulative_returns['Test'].index[-1]) # Ensures that the x-axis ends at the last date

# Set x-axis formatter
ax.xaxis.set_major_formatter(CustomDateFormatter())            

# Optionally set the title
ax.set_title(f'Cumulative Returns across data splits $~\mid~ (L={L},~\\theta = \\lfloor {prop_of_k}k \\rfloor)$ ', fontsize=TITLE_FONTSIZE, pad=50) if not save_this_plot else None

# Add legend
ax.legend(handles=legend_elements, fontsize=LEGEND_FONTSIZE)

# Save the plot if requested
plt.savefig(os.path.join(path_output, f'{model}_Portfolio_Cum_Returns.pdf'), bbox_inches='tight') if save_this_plot else None

# Show the plot
plt.show()


# 
# ---
# ---

# Distribution of Cluster-Average Sharpe Ratios $\overline{SR}_g$

save_this_plot  = True

#===============================================================================================================

# Prepare data for plotting
data = []
for (split, cluster), avg_sr in SR_Average_dict.items():
    data.append((split, avg_sr))

# Create a DataFrame for easier plotting with seaborn
df_sr = pd.DataFrame(data, columns=['Split', 'Average_SR'])

# Set the plotting style
sns.set(style="whitegrid")

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Initialize the figure
plt.figure(figsize=(14, 8))

# Create a density plot for the distribution of Sharpe Ratios across each split
# Use a loop to ensure labels are set for the legend
splits = df_sr['Split'].unique()
for split in splits:
    subset = df_sr[df_sr['Split'] == split]
    sns.kdeplot(subset['Average_SR'], label=split, fill=True, common_norm=False)

# Set y-axis limits
plt.ylim(0, 0.21)

# Customize the plot
plt.title('Distribution of Cluster-Average Sharpe Ratios by Split', fontsize=TITLE_FONTSIZE, pad=TITLE_PAD) if not save_this_plot else None
plt.xlabel('Cluster-Average Sharpe Ratio ($\\overline{SR}_g$)', fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
plt.ylabel('Density', fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
plt.xticks(fontsize=TICK_FONTSIZE)
plt.yticks(fontsize=TICK_FONTSIZE)

plt.xticks(range(-30, 31, 5))  # Set x-ticks from -30 to 30 with step of 5
plt.xlim(-22, 22)  # Set x-axis limits from -30 to 30

plt.grid(True, alpha=0.3, linestyle=':')
plt.gca().set_facecolor('#f5f5f5')  # Set the background color of the plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.axvline(x=0, color='black', linestyle='--', linewidth=2)

# Add the legend with the specified font size
plt.legend(title='Split', fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE)

plt.savefig(os.path.join(path_output, f'fig_{model}_Cluster-Avg_SR_Distribution.pdf'), bbox_inches='tight') if save_this_plot else None
# Show the plot
plt.show()


# 
# ---
# ---

# Robustness Checks

# 1) Holding Period Length $(L)$

P_statistics_dict_L = {}
prop_of_k = 0.5
θ = int(prop_of_k*k_opt)    # Number of Traded clusters
grid_L = range(1,101)  # Define the grid of holding periods

def compute_P_statistics(L):
    print('=*'*80)
    print(f'Processing {L=}')
    print('=*'*80)

    SR_split_cluster_dict = {}  # Initialize dictionary to store sum of SR and count of observations for each cluster and split

    for idx_row, row in B.iterrows():
        TS_data = TS_dict[idx_row]
        if TS_data is not None and isinstance(TS_data, pd.DataFrame) and len(TS_data) > L:
            split = row['split']
            cluster = row['cluster']
            SR_L = TS_data['SR'][L]

            # Create dictionary keys if they don't exist
            if (split, cluster) not in SR_split_cluster_dict:
                SR_split_cluster_dict[(split, cluster)] = {'SR_sum': 0, 'count': 0}

            # Accumulate SR values and count
            SR_split_cluster_dict[(split, cluster)]['SR_sum'] += SR_L  # Accumulate the SR values
            SR_split_cluster_dict[(split, cluster)]['count'] += 1  # Increment the count

    SR_split_cluster_dict = OrderedDict(sorted(SR_split_cluster_dict.items(), key=lambda x: x[0][1]))  # Sort the dictionary by cluster number

    SR_Average_dict = {}  # Initialize dictionary to store average SR for each cluster and split
    for (split, cluster), values in SR_split_cluster_dict.items():
        if values['count'] > 0:
            SR_Average_dict[(split, cluster)] = values['SR_sum'] / values['count']  # Compute the average SR
        else:
            SR_Average_dict[(split, cluster)] = float('nan')  # or 0 if you prefer

    SR_split_dict = {}  # Group the average SRs by splits
    for (split, cluster), avg_sr in SR_Average_dict.items():
        if split not in SR_split_dict:
            SR_split_dict[split] = []
        SR_split_dict[split].append((cluster, avg_sr))

    SR_Ranking_dict = {}  # Sort and rank the SRs within each split
    for split, sr_list in SR_split_dict.items():
        SR_Ranking_dict[split] = sorted(sr_list, key=lambda x: x[1], reverse=True)

    #================ GREEDY ALGORITHM ================#
    # Initialize the sets for clusters
    G_SR_plus_train = set()
    G_SR_minus_train = set()
    G_SR_plus_val = set()
    G_SR_minus_val = set()
    G_SR_plus_test = set()
    G_SR_minus_test = set()

    # Iterate through the ranked SRs and classify the clusters
    for split, ranked_sr in SR_Ranking_dict.items():
        for cluster, avg_sr in ranked_sr:
            if split == 'Train':
                if avg_sr > 0:
                    G_SR_plus_train.add(cluster)
                else:
                    G_SR_minus_train.add(cluster)
            elif split == 'Validation':
                if avg_sr > 0:
                    G_SR_plus_val.add(cluster)
                else:
                    G_SR_minus_val.add(cluster)
            elif split == 'Test':
                if avg_sr > 0:
                    G_SR_plus_test.add(cluster)
                else:
                    G_SR_minus_test.add(cluster)

    # Sort clusters in the validation sample by average SR
    sorted_positive_val = sorted(G_SR_plus_val, key=lambda cluster: SR_Average_dict[('Validation', cluster)], reverse=True)
    sorted_negative_val = sorted(G_SR_minus_val, key=lambda cluster: SR_Average_dict[('Validation', cluster)], reverse=False)

    # Select the top θ positions, upper bounded by the cardinality of the G_SR sets
    G_θ_plus = sorted_positive_val[:min(θ, len(sorted_positive_val))]
    G_θ_minus = sorted_negative_val[:min(θ, len(sorted_negative_val))]

    B['TR_Greedy'] = 0.0
    B.loc[B['cluster'].isin(G_θ_plus), 'TR_Greedy'] = 1.0
    B.loc[B['cluster'].isin(G_θ_minus), 'TR_Greedy'] = -1.0

    #================ STABLE ALGORITHM ================#
    Split1 = 'Train'
    Split2 = 'Validation'

    # Calculate the ranks for each cluster in each split
    ranks = {split: {cluster: rank for rank, (cluster, _) in enumerate(ranked_sr, start=1)}
            for split, ranked_sr in SR_Ranking_dict.items()}

    # Extract the common clusters across Split1 and Split2
    common_clusters = set(ranks[Split1]).intersection(ranks[Split2])

    # Calculate Spearman rank correlation coefficient for the ranks between Split1 and Split2
    split1_ranks = []
    split2_ranks = []
    for cluster in common_clusters:
        split1_ranks.append(ranks[Split1][cluster])
        split2_ranks.append(ranks[Split2][cluster])

    rank_differences = {cluster: abs(ranks[Split1][cluster] - ranks[Split2][cluster]) for cluster in common_clusters}  # Calculate the absolute rank differences between Split1 and Split2
    sorted_rank_differences = sorted(rank_differences.items(), key=lambda x: x[1])  # Sort clusters by the absolute rank differences
    most_stable_clusters = [cluster for cluster, _ in sorted_rank_differences[:2*θ]]  # Select the top θ clusters with the smallest rank differences

    # Determine long and short positions based on average Sharpe Ratios
    long_clusters = [cluster for cluster in most_stable_clusters if SR_Average_dict[(Split1, cluster)] > 0 and SR_Average_dict[(Split2, cluster)] > 0]
    short_clusters = [cluster for cluster in most_stable_clusters if SR_Average_dict[(Split1, cluster)] < 0 and SR_Average_dict[(Split2, cluster)] < 0]

    B['TR_RankStable'] = 0.0
    B.loc[B['cluster'].isin(long_clusters), 'TR_RankStable'] = 1.0
    B.loc[B['cluster'].isin(short_clusters), 'TR_RankStable'] = -1.0

    TradingRule_dict = {
        'Greedy': 'TR_Greedy',
        'Stable': 'TR_RankStable'
    }

    # Create a loop to calculate the portfolio returns for each Trading Rule, storing in the same dataframe r_P_all for both Trading Rules, r_P_train for both Trading Rules, r_P_val for both Trading Rules, and r_P_test for both Trading Rules
    r_P_all, r_P_train, r_P_val, r_P_test = {}, {}, {}, {}

    for key, TR in TradingRule_dict.items():
        r_P_dict = calculate_portfolio_returns(B, 𝖉, L, TS_dict, TradingRule=TR, verbose=False)
        r_P_all[key] = r_P_dict['All']['returns']
        r_P_train[key] = r_P_dict['Train']['returns']
        r_P_val[key] = r_P_dict['Validation']['returns']
        r_P_test[key] = r_P_dict['Test']['returns']

    # Convert into a dataframe
    del r_P_dict
    r_P_all, r_P_train, r_P_val, r_P_test = pd.DataFrame(r_P_all), pd.DataFrame(r_P_train), pd.DataFrame(r_P_val), pd.DataFrame(r_P_test)
    r_P_dict = {'All': r_P_all, 'Train': r_P_train, 'Validation': r_P_val, 'Test': r_P_test}

    P_statistics = {'All': {}, 'Train': {}, 'Validation': {}, 'Test': {}}

    # Loop to apply portfolio_statistics function to each dataframe in r_P_dict
    for split, r_P in r_P_dict.items():
        for TR_key in TradingRule_dict.keys():
            r_P_series = r_P[TR_key]
            cum_r_P_final, μ_P, σ_P, SR_P = portfolio_statistics(r_P_series)
            P_statistics[split][TR_key] = {
                'cum_r_P_final': cum_r_P_final,
                'μ_P': μ_P,
                'σ_P': σ_P,
                'SR_P': SR_P
            }

    return L, P_statistics


# Obtain the ``P_statistics`` for a grid of $L$

# Use joblib to parallelize the computation
results_L = Parallel(n_jobs=-1)(delayed(compute_P_statistics)(L) for L in grid_L)

# Collect the results
P_statistics_dict_L = {L: P_statistics for L, P_statistics in results_L}

# - The robustness checks are done in the **Test** sample
# - However, we also use this code to justify the choice of hyperparameters ($L=4$, $\theta = \lfloor 0.5k \rfloor$). In this case, we set `Split` = 'Train' and 'Validation'

#===================================================

Split = 'Train'

#===================================================

# Extract the series of Sharpe Ratios in the test set for each L
SR_test_set = {L: P_statistics_dict_L[L][Split] for L in grid_L}

# Extract Sharpe Ratios for each trading rule in the test set
SR_series_test = {L: {tr_key: stats['SR_P'] for tr_key, stats in SR_test_set[L].items()} for L in grid_L}

# Cnvert it to a DataFrame for easier   plotting
SR_df = pd.DataFrame(SR_series_test).T

# Select the first 50 L values
SR_subset = SR_df.head(20)

print(f'{SR_df.shape     = }')
print(f'{SR_subset.shape = }')

# Distribution or $SR$ in the Test Set

save_this_plot = True
show_title = True

#===============================================================================================================

# Set the plotting style
sns.set(style="whitegrid")

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

colors = {'Greedy': 'blue', 'Stable': 'green'}

# Initialize the figure
plt.figure(figsize=(14, 8))

# Create a density plot for the distribution of Sharpe Ratios across each column
# Use a loop to ensure labels are set for the legend
for column in SR_subset.columns:
    sns.kdeplot(SR_subset[column], label=column, fill=True, common_norm=False, color=colors[column])

# Customize the plot
plt.title('Distribution of Sharpe Ratios in the Test set for Different selection algorithms', fontsize=TITLE_FONTSIZE, pad=TITLE_PAD) if show_title else None
plt.xlabel(f'Sharpe Ratio ({Split})', fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
plt.ylabel('Density', fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
plt.xticks(fontsize=TICK_FONTSIZE)
plt.yticks(fontsize=TICK_FONTSIZE)

plt.axvline(x=0, color='black', linestyle='--', linewidth=2)

plt.grid(True, alpha=0.3, linestyle=':')
plt.gca().set_facecolor('#f5f5f5')  # Set the background color of the plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Add the legend with the specified font size
plt.legend(fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE, loc='best')

plt.savefig(os.path.join(path_output, f'{model}_RobustnessCheck_SR_Distribution_[{Split}]_[Change_L].pdf'), bbox_inches='tight') if save_this_plot else None

# Show the plot
plt.show()


save_this_plot = True
show_title = True

#===============================================================================================================

# Set the plotting style
sns.set(style="whitegrid")

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

colors = {'Greedy': 'blue', 'Stable': 'green'}

# Initialize the figure
plt.figure(figsize=(14, 8))

# Plot the Sharpe Ratios for each L
for column in SR_subset.columns:
    plt.plot(SR_subset.index, SR_subset[column], label=column, color=colors[column])

# Customize the plot
plt.title('Sharpe Ratio of the Portfolio for Different Holding Periods ($L$)', fontsize=TITLE_FONTSIZE, pad=TITLE_PAD) if show_title else None
plt.xlabel('Holding period length ($L$)', fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
plt.ylabel(f'Sharpe Ratio ({Split})', fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
plt.xticks(fontsize=TICK_FONTSIZE)
plt.yticks(fontsize=TICK_FONTSIZE)
plt.grid(True, alpha=0.3, linestyle=':')
plt.gca().set_facecolor('#f5f5f5')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

xticks = list(range(2, len(SR_subset)+1, 2))  # Set xticks every 10
plt.xticks(xticks)
# ticks should not end in .0
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

plt.axvline(x=4, color='red', linestyle='--', linewidth=2)     # plot a vertical line at L=4
plt.axhline(y=0, color='black', linestyle='--', linewidth=2)     # Add a horizontal line at 0

# Add a legend with larger font size
plt.legend(fontsize=LEGEND_FONTSIZE, loc='best')

plt.savefig(os.path.join(path_output, f'{model}_RobustnessCheck_SR_vs_L_[{Split}]_[Change_L].pdf'), bbox_inches='tight') if save_this_plot else None

# Show the plot
plt.show()


# 
# ---

# 2) Number of Traded Clusters $(\theta)$

P_statistics_dict = {}
grid_θ = range(1, 15)  # Define the grid of number of traded clusters
L = 4

from joblib import Parallel, delayed

def compute_P_statistics_θ(θ):
    print('=*'*80)
    print(f'Processing {θ=}')
    print('=*'*80)

    SR_split_cluster_dict = {}  # Initialize dictionary to store sum of SR and count of observations for each cluster and split

    for idx_row, row in B.iterrows():
        TS_data = TS_dict[idx_row]
        if TS_data is not None and isinstance(TS_data, pd.DataFrame) and len(TS_data) > L:
            split = row['split']
            cluster = row['cluster']
            SR_L = TS_data['SR'][L]

            # Create dictionary keys if they don't exist
            if (split, cluster) not in SR_split_cluster_dict:
                SR_split_cluster_dict[(split, cluster)] = {'SR_sum': 0, 'count': 0}

            # Accumulate SR values and count
            SR_split_cluster_dict[(split, cluster)]['SR_sum'] += SR_L  # Accumulate the SR values
            SR_split_cluster_dict[(split, cluster)]['count'] += 1  # Increment the count

    SR_split_cluster_dict = OrderedDict(sorted(SR_split_cluster_dict.items(), key=lambda x: x[0][1]))  # Sort the dictionary by cluster number

    SR_Average_dict = {}  # Initialize dictionary to store average SR for each cluster and split
    for (split, cluster), values in SR_split_cluster_dict.items():
        if values['count'] > 0:
            SR_Average_dict[(split, cluster)] = values['SR_sum'] / values['count']  # Compute the average SR
        else:
            SR_Average_dict[(split, cluster)] = float('nan')  # or 0 if you prefer

    SR_split_dict = {}  # Group the average SRs by splits
    for (split, cluster), avg_sr in SR_Average_dict.items():
        if split not in SR_split_dict:
            SR_split_dict[split] = []
        SR_split_dict[split].append((cluster, avg_sr))

    SR_Ranking_dict = {}  # Sort and rank the SRs within each split
    for split, sr_list in SR_split_dict.items():
        SR_Ranking_dict[split] = sorted(sr_list, key=lambda x: x[1], reverse=True)

    #================ GREEDY ALGORITHM ================#
    # Initialize the sets for clusters
    G_SR_plus_train = set()
    G_SR_minus_train = set()
    G_SR_plus_val = set()
    G_SR_minus_val = set()
    G_SR_plus_test = set()
    G_SR_minus_test = set()

    # Iterate through the ranked SRs and classify the clusters
    for split, ranked_sr in SR_Ranking_dict.items():
        for cluster, avg_sr in ranked_sr:
            if split == 'Train':
                if avg_sr > 0:
                    G_SR_plus_train.add(cluster)
                else:
                    G_SR_minus_train.add(cluster)
            elif split == 'Validation':
                if avg_sr > 0:
                    G_SR_plus_val.add(cluster)
                else:
                    G_SR_minus_val.add(cluster)
            elif split == 'Test':
                if avg_sr > 0:
                    G_SR_plus_test.add(cluster)
                else:
                    G_SR_minus_test.add(cluster)

    # Sort clusters in the validation sample by average SR
    sorted_positive_val = sorted(G_SR_plus_val, key=lambda cluster: SR_Average_dict[('Validation', cluster)], reverse=True)
    sorted_negative_val = sorted(G_SR_minus_val, key=lambda cluster: SR_Average_dict[('Validation', cluster)], reverse=False)

    # Select the top θ positions, upper bounded by the cardinality of the G_SR sets
    G_θ_plus = sorted_positive_val[:min(θ, len(sorted_positive_val))]
    G_θ_minus = sorted_negative_val[:min(θ, len(sorted_negative_val))]
    print(f'{G_θ_plus = }')
    print(f'{G_θ_minus = }')

    B['TR_Greedy'] = 0
    B.loc[B['cluster'].isin(G_θ_plus), 'TR_Greedy'] = 1
    B.loc[B['cluster'].isin(G_θ_minus), 'TR_Greedy'] = -1

    #================ STABLE ALGORITHM ================#
    Split1 = 'Train'
    Split2 = 'Validation'

    # Calculate the ranks for each cluster in each split
    ranks = {split: {cluster: rank for rank, (cluster, _) in enumerate(ranked_sr, start=1)}
            for split, ranked_sr in SR_Ranking_dict.items()}

    # Extract the common clusters across Split1 and Split2
    common_clusters = set(ranks[Split1]).intersection(ranks[Split2])

    # Calculate Spearman rank correlation coefficient for the ranks between Split1 and Split2
    split1_ranks = []
    split2_ranks = []
    for cluster in common_clusters:
        split1_ranks.append(ranks[Split1][cluster])
        split2_ranks.append(ranks[Split2][cluster])

    rank_differences = {cluster: abs(ranks[Split1][cluster] - ranks[Split2][cluster]) for cluster in common_clusters}  # Calculate the absolute rank differences between Split1 and Split2
    sorted_rank_differences = sorted(rank_differences.items(), key=lambda x: x[1])  # Sort clusters by the absolute rank differences
    most_stable_clusters = [cluster for cluster, _ in sorted_rank_differences[:2*θ]]  # Select the top θ clusters with the smallest rank differences

    # Determine long and short positions based on average Sharpe Ratios
    long_clusters = [cluster for cluster in most_stable_clusters if SR_Average_dict[(Split1, cluster)] > 0 and SR_Average_dict[(Split2, cluster)] > 0]
    short_clusters = [cluster for cluster in most_stable_clusters if SR_Average_dict[(Split1, cluster)] < 0 and SR_Average_dict[(Split2, cluster)] < 0]

    print(f'{long_clusters = }')
    print(f'{short_clusters = }')

    B['TR_RankStable'] = 0
    B.loc[B['cluster'].isin(long_clusters), 'TR_RankStable'] = 1
    B.loc[B['cluster'].isin(short_clusters), 'TR_RankStable'] = -1

    TradingRule_dict = {
        'Greedy': 'TR_Greedy',
        'Stable': 'TR_RankStable'
    }

    # Create a loop to calculate the portfolio returns for each Trading Rule, storing in the same dataframe r_P_all for both Trading Rules, r_P_train for both Trading Rules, r_P_val for both Trading Rules, and r_P_test for both Trading Rules
    r_P_all, r_P_train, r_P_val, r_P_test = {}, {}, {}, {}

    for key, TR in TradingRule_dict.items():
        r_P_dict = calculate_portfolio_returns(B, 𝖉, L, TS_dict, TradingRule=TR, verbose=False)
        r_P_all[key] = r_P_dict['All']['returns']
        r_P_train[key] = r_P_dict['Train']['returns']
        r_P_val[key] = r_P_dict['Validation']['returns']
        r_P_test[key] = r_P_dict['Test']['returns']

    # Convert into a dataframe
    del r_P_dict
    r_P_all, r_P_train, r_P_val, r_P_test = pd.DataFrame(r_P_all), pd.DataFrame(r_P_train), pd.DataFrame(r_P_val), pd.DataFrame(r_P_test)
    r_P_dict = {'All': r_P_all, 'Train': r_P_train, 'Validation': r_P_val, 'Test': r_P_test}

    P_statistics = {'All': {}, 'Train': {}, 'Validation': {}, 'Test': {}}

    # Loop to apply portfolio_statistics function to each dataframe in r_P_dict
    for split, r_P in r_P_dict.items():
        for TR_key in TradingRule_dict.keys():
            r_P_series = r_P[TR_key]
            cum_r_P_final, μ_P, σ_P, SR_P = portfolio_statistics(r_P_series)
            P_statistics[split][TR_key] = {
                'cum_r_P_final': cum_r_P_final,
                'μ_P': μ_P,
                'σ_P': σ_P,
                'SR_P': SR_P
            }

    return θ, P_statistics



# Obtain the ``P_statistics`` for a grid of $\theta$

# Use joblib to parallelize the computation
results_θ = Parallel(n_jobs=-1)(delayed(compute_P_statistics_θ)(θ) for θ in grid_θ)

# Collect the results
P_statistics_dict_θ = {θ: P_statistics for θ, P_statistics in results_θ}

# Extract the series of Sharpe Ratios in the test set for each L
SR_test_set = {θ: P_statistics_dict_θ[θ][Split] for θ in grid_θ}

# Extract Sharpe Ratios for each trading rule in the test set
SR_series_test = {θ: {tr_key: stats['SR_P'] for tr_key, stats in SR_test_set[θ].items()} for θ in grid_θ}

# Cnvert it to a DataFrame for easier plotting
SR_df = pd.DataFrame(SR_series_test).T

print(f'{SR_df.shape     = }')

# Distribution or $SR$ in the Test Set

save_this_plot = True
show_title = True

#===============================================================================================================

# Set the plotting style
sns.set(style="whitegrid")

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

colors = {'Greedy': 'blue', 'Stable': 'green'}

# Initialize the figure
plt.figure(figsize=(14, 8))

# Create a density plot for the distribution of Sharpe Ratios across each column
# Use a loop to ensure labels are set for the legend
for column in SR_df.columns:
    sns.kdeplot(SR_df[column], label=column, fill=True, common_norm=False, color=colors[column])

# Customize the plot
plt.title('Distribution of Sharpe Ratios in the Test set for Different selection algorithms', fontsize=TITLE_FONTSIZE, pad=TITLE_PAD) if show_title else None
plt.xlabel(f'Sharpe Ratio ({Split})', fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
plt.ylabel('Density', fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
plt.xticks(fontsize=TICK_FONTSIZE)
plt.yticks(fontsize=TICK_FONTSIZE)

plt.axvline(x=0, color='black', linestyle='--', linewidth=2)

plt.grid(True, alpha=0.3, linestyle=':')
plt.gca().set_facecolor('#f5f5f5')  # Set the background color of the plot
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Add the legend with the specified font size
plt.legend(fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE, loc='best')

plt.savefig(os.path.join(path_output, f'{model}_RobustnessCheck_SR_Set_Distribution_[{Split}]_[Change_theta].pdf'), bbox_inches='tight') if save_this_plot else None

# Show the plot
plt.show()


save_this_plot = True
show_title = True
#===============================================================================================================

# Set the plotting style
sns.set(style="whitegrid")

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

colors = {'Greedy': 'blue', 'Stable': 'green'}

# Initialize the figure
plt.figure(figsize=(14, 8))

# Plot the Sharpe Ratios for each L
for column in SR_df.columns:
    plt.plot(SR_df.index, SR_df[column], label=column, color=colors[column])

# Customize the plot
plt.title(r'Sharpe Ratio in the Test Split for Different Number of Traded Clusters ($\theta$)', fontsize=TITLE_FONTSIZE, pad=TITLE_PAD) if show_title else None
plt.xlabel(r'Number of traded clusters ($\theta$)', fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
plt.ylabel(f'Sharpe Ratio ({Split})', fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
plt.xticks(fontsize=TICK_FONTSIZE)
plt.yticks(fontsize=TICK_FONTSIZE)
plt.grid(True, alpha=0.3, linestyle=':')
plt.gca().set_facecolor('#f5f5f5')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.xticks(range(1, len(SR_df)+1, 1))

plt.axvline(x=9, color='red', linestyle='--', linewidth=2)     # plot a vertical line at L=4
plt.axhline(y=0, color='black', linestyle='--', linewidth=2)     # Add a horizontal line at 0

# Add a legend with larger font size
plt.legend(fontsize=LEGEND_FONTSIZE, loc='best')

plt.savefig(os.path.join(path_output, f'{model}_RobustnessCheck_SR_Set_vs_Theta_[{Split}]_[Change_theta].pdf'), bbox_inches='tight') if save_this_plot else None

# Show the plot
plt.show()


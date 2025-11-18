import matplotlib.pyplot as plt
from src import get_logger
from src.config import config_settings

# IN THIS FILE, WE SHOULD SET UP THE CREATION OF THE PLOTTING-RELATED FOLDERS
# EACH FOLDER SHOULD HAVE ITS OWN SPECIFIC DIRECTORY CREATION CONFIGURATION 
# REMOVE THE AUTOMATIC GENERATION OF ALL FOLDERS FROM CONFIG, AND ONLY CREATE THE DATA FOLDERS THERE
# THEN, IN PLOTTING, TABLING, AND SO ON, WE SHOULD CREATE THE FOLDERS AS NEEDED


def setup_matplotlib_config():
    """Configure matplotlib using settings from config_settings.py"""
    logger = get_logger(__name__)
    
    # Apply LaTeX and font settings
    tex_settings = config_settings.plotting.get('tex_settings', {})
    for key, value in tex_settings.items():
        plt.rcParams[key] = value
    
    # Apply font sizes
    fontsize = config_settings.plotting.get('fontsize', {})
    plt.rcParams['axes.titlesize'] = fontsize.get('title_size', 24)
    plt.rcParams['axes.labelsize'] = fontsize.get('label_size', 20)
    plt.rcParams['xtick.labelsize'] = fontsize.get('tick_size', 16)
    plt.rcParams['ytick.labelsize'] = fontsize.get('tick_size', 16)
    plt.rcParams['legend.fontsize'] = fontsize.get('legend_size', 18)

    # Apply figure settings
    figsize = config_settings.plotting.get('figsize', {})
    plt.rcParams['figure.figsize'] = [figsize.get('width', 15), figsize.get('height', 10)]

    # Apply color settings
    colors = config_settings.plotting.get('colors', {})
    if 'background' in colors:
        plt.rcParams['axes.facecolor'] = colors['background']
        
    # Apply padding (these are used when calling plt.title/plt.xlabel/plt.ylabel)
    padding = config_settings.plotting.get('padding', {})
    plt.rcParams['axes.titlepad'] = padding.get('title_pad', 15)
    plt.rcParams['axes.labelpad'] = padding.get('label_pad', 7)

    logger.debug("✔️ Matplotlib configured from config_settings.py")

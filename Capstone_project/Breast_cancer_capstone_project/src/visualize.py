import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib import rcParams

# Global configuration
PLOT_SAVE_DIR = "results/plots"
DISPLAY_PLOTS = False  # Set to True to show plots interactively

def set_plot_style(save_mode=False):
    """Set custom plotting style for all visualizations"""
    plt.style.use('default')
    config = {
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.grid': False,
        'xtick.bottom': True,
        'xtick.labelbottom': True,
        'xtick.direction': 'out',
        'ytick.left': True,
        'font.size': 12,
        'savefig.dpi': 300,  # High resolution
        'savefig.bbox': 'tight',  # Tight layout
        'savefig.format': 'png'  # Default format
    }
    
    if save_mode:
        config.update({
            'interactive': False  # Disable interactive mode when saving
        })
    
    rcParams.update(config)

def ensure_plot_dir():
    """Ensure plot directory exists"""
    os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

def save_or_show(fig, filename):
    """
    Handle plot display/saving based on global settings
    Args:
        fig: matplotlib figure object
        filename: name for saved file (without extension)
    """
    ensure_plot_dir()
    save_path = os.path.join(PLOT_SAVE_DIR, f"{filename}.png")
    
    fig.savefig(save_path)
    print(f"Plot saved to {save_path}")
    
    if DISPLAY_PLOTS:
        plt.show()
    else:
        plt.close(fig)

def plot_feature_distributions(df, exclude_cols=None, save_file="feature_distributions"):
    """
    Plot histogram distributions for all numerical features
    
    Args:
        df: DataFrame containing the data
        exclude_cols: List of columns to exclude from plotting
        save_file: Base filename for saving (without extension)
    """
    set_plot_style(save_mode=True)
    
    if exclude_cols is None:
        exclude_cols = []
    
    # Select columns to plot
    columns_to_plot = [col for col in df.select_dtypes(include=['number']).columns 
                      if col not in exclude_cols]
    
    # Calculate layout
    n_rows = 3
    n_cols = (len(columns_to_plot) + n_rows - 1) // n_rows
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12))
    fig.suptitle('Feature Distributions', fontsize=16, y=1.02)
    axes = axes.flatten()
    
    # Plot histograms
    colors = plt.cm.tab10(np.linspace(0, 1, len(columns_to_plot)))
    for i, col in enumerate(columns_to_plot):
        ax = axes[i]
        df[col].hist(ax=ax, bins=np.arange(1, 12)-0.5,
                    color=colors[i], edgecolor='white',
                    linewidth=1.2, alpha=0.85)
        
        ax.set_title(col, pad=12, fontsize=13, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_xticks(range(1, 11))
        ax.set_xlim(0.5, 10.5)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.8)
        ax.grid(False)
    
    # Clean up empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)
    save_or_show(fig, save_file)

def plot_correlation_heatmap(df, target_col='class', save_file="correlation_heatmap"):
    """
    Plot correlation heatmap and return top correlated features
    
    Args:
        df: DataFrame containing the data
        target_col: Name of the target column
        save_file: Base filename for saving (without extension)
    
    Returns:
        List of top 5 features correlated with target
    """
    set_plot_style(save_mode=True)
    
    cor_data = df.select_dtypes(include=['number'])
    cor_mat = cor_data.corr()
    
    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(cor_mat, cmap='coolwarm', annot=True, fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    save_or_show(fig, save_file)
    
    if target_col in cor_mat.columns:
        return cor_mat[target_col].sort_values(
            ascending=False).abs().index[1:6].to_list()
    return []

def plot_feature_relationships(df, features, target_col='class', save_file="feature_relationships"):
    """
    Plot pairwise relationships between features
    
    Args:
        df: DataFrame containing the data
        features: List of features to include in pairplot
        target_col: Name of the target column for hue
        save_file: Base filename for saving (without extension)
    """
    set_plot_style(save_mode=True)
    
    if target_col in df.columns:
        plot = sns.pairplot(df[features + [target_col]], hue=target_col)
    else:
        plot = sns.pairplot(df[features])
    
    plot.figure.suptitle('Feature Relationships', y=1.02)
    save_or_show(plot.figure, save_file)
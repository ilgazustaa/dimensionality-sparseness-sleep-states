### Define a few global variables
import os
import numpy as np
import matplotlib.pyplot as plt

# Path for data, figures
project_path = "/home/friedrich/code/mesoscale_project/"
data_path_sup = os.path.join(project_path, "data/suppl_fig_model")
path_data_simulation = os.path.join(data_path_sup, "simulation_data")
path_data_simulation_analysis = os.path.join(data_path_sup, "simulation_analyzed")
path_data_experiment = os.path.join(data_path_sup, "experiment_data")
path_data_experiment_analysis = os.path.join(data_path_sup, "experiment_analyzed")
figure_path = os.path.join(project_path, "suppl_fig_model/figures")

# Colors
# Set Set1 for all qualitatve colors, and as the default color palette. 
import seaborn as sns
colors = sns.color_palette("Set1")
sns.set_palette(colors)

# Specific colors
cmap_rates = "Greys"
cmap_corr = "PiYG"
# EI: blue for I, dark red for E.
c_I = "#A64800FD"
c_E = plt.cm.Reds(1/1.1)
cs_EI = np.array(["#A64800FD", "#1a679e"])
# Colors for input and gain modulation.
c_in = colors[2]
c_gm = colors[3]
# Colormaps for weights
from matplotlib.colors import ListedColormap
cmap_E = ListedColormap(['w', cs_EI[0]])
cmap_I = ListedColormap(['w', cs_EI[1]])
# cmap_EI = ListedColormap([cs_EI[1], 'w', cs_EI[0]])
cmap_EI = "seismic"


### Plotting
# Fontsize
fontsize = 9
fs = fontsize # Short-hand in scripts.
# Figure labels
flbs = ["(%s)"%s for s in "abcdefghijkl"] 
flbs = ["%s"%s for s in "ABCDEFGHIJKL"] 
# Color for legend
c_leg = '0.7'

# Axis colors and font family
axes_color = "#959595" 
text_color = "#363636"
font_family = 'sans-serif'
linewidth  = 2
lines_markersize = 3
axes_linewidth = 0.5

# Figure size
# Default height is defined by the golden ratio. 
from  scipy.constants import golden_ratio as gr
fig_width = 3.5 # inch
fig_height = fig_width / gr  
figsize  = (fig_width, fig_height)
# Large, 2-column figures.
fig_width_large = 7.3 # inch
fig_height_large = fig_width_large / gr  


def set_mpl_rc():
    import matplotlib.pyplot as plt
    # from matplotlib.colors import colorConverter
    from matplotlib import rcParams
    try:
        import seaborn as sns
        sns.set(style='ticks', palette='Set1') 
    except:
        print('seaborn not installed')
        
    # Set default colors
    from cycler import cycler
    plt.rc('axes', prop_cycle=(cycler('color', colors)))
    
    
    fontsize_labels         = fontsize    # pt, size used in latex document
    fontsize_labels_axes    = fontsize_labels
    fontsize_labels_title   = fontsize_labels
    fontsize_plotlabel      = fontsize_labels       # for labeling plots with 'A', 'B', etc.
    legend_ms  = 2  # scale of markers in legend
    
    # Adapt the matplotlib.rc
    rcParams['font.family']         = font_family
    rcParams['text.usetex']         = False
    rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' #for \text command
    rcParams['figure.figsize']      = figsize
    rcParams['font.weight']         = "light"
    rcParams['font.size']           = fontsize_labels
    rcParams['xtick.labelsize']     = fontsize_labels
    rcParams['ytick.labelsize']     = fontsize_labels
    rcParams['legend.fontsize']     = fontsize_labels
    rcParams['axes.labelsize']      = fontsize_labels_axes
    rcParams['axes.titlesize']      = fontsize_labels_title
    rcParams['legend.markerscale']  = legend_ms
    rcParams['text.color']          = text_color
    rcParams['xtick.color']         = text_color
    rcParams['ytick.color']         = text_color
    rcParams['axes.labelcolor']     = text_color
    rcParams['axes.edgecolor']      = axes_color
    rcParams['axes.linewidth']      = axes_linewidth
    rcParams['axes.grid']           = False
    rcParams['lines.linewidth']     = linewidth
    rcParams['lines.markersize']    = lines_markersize
    
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    
    tick_params = {
                    ## TICKS
                    # see http://matplotlib.org/api/axis_api.html#matplotlib.axis.Tick
                    'xtick.major.size'     : 3,      # major tick size in points
                    'xtick.minor.size'     : 2,      # minor tick size in points
                    'xtick.major.width'    : 0.5,    # major tick width in points
                    'xtick.minor.width'    : 0.5,    # minor tick width in points
                    'xtick.major.pad'      : 4,      # distance to major tick label in points
                    'xtick.minor.pad'      : 4,      # distance to the minor tick label in points
                    'xtick.direction'      : 'out',    # direction: in, out, or inout
                    
                    'ytick.major.size'     : 3,      # major tick size in points
                    'ytick.minor.size'     : 2,      # minor tick size in points
                    'ytick.major.width'    : 0.5,    # major tick width in points
                    'ytick.minor.width'    : 0.5,    # minor tick width in points
                    'ytick.major.pad'      : 4,      # distance to major tick label in points
                    'ytick.minor.pad'      : 4,      # distance to the minor tick label in points
                    'ytick.direction'      : 'out'    # direction: in, out, or inout
                    }
    rcParams.update(tick_params)

    return None
set_mpl_rc()

def scale_plot_size(factor=2):
    import matplotlib as mpl
    default_dpi = mpl.rcParamsDefault['figure.dpi']
    mpl.rcParams['figure.dpi'] = default_dpi*factor


def color_iterator(n_iter, cmap='viridis', vmin=0):
    """ Continuous line coloring. 
    n_iter: int
        Number of color levels = iterations.
    cmap: str
        Color map. Check matplotlib for possible values. 
        default: 'viridis'
        
    Returns: 
    scalarMap:
        Generator for colors. Usage:
        ```
            from helper_funcs import color_iterator
            c_iter = color_iterator(n_iter)
            ax.plot(x, y, c=c_iter(idx))
        ```    
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mplcolors
    import matplotlib.cm as cmx
    cm          = plt.get_cmap(cmap) 
    cNorm       = mplcolors.Normalize(vmin=vmin, vmax=n_iter-1)
    scalarMap   = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    c_iter = lambda idx: scalarMap.to_rgba(idx)
    return c_iter


def saving_fig(fig, figure_path, fig_name, data_type="png", verbose=True, dpi=1000, save_fig=True):
    ### Helper function to save figures
    # DPI = 1000 is a simple recommendation for publication plots
    from pathlib import Path
    if not Path(figure_path).is_dir():
        os.makedirs(figure_path)
        print("Made new directory ", figure_path)        
    if save_fig:
        if verbose:
            print("Save figure to " + os.path.join(figure_path, fig_name) + "." + data_type)
        if data_type in ("png", "both"):
            fig.savefig(os.path.join(figure_path, fig_name + ".png"), 
                    dpi=dpi, 
                    bbox_inches='tight', format="png") 
        if data_type in ("pdf", "both"):
            fig.savefig(os.path.join(figure_path, fig_name + ".pdf"), 
                    dpi=dpi, 
                    bbox_inches='tight', format="pdf")
        if not data_type in ("png", "pdf", "both"):
            fig.savefig(os.path.join(figure_path, fig_name + "." + data_type),
                    format=data_type)
    else:
        if verbose:
            print("Not saved. Path would be: " + os.path.join(figure_path, fig_name) + "." + data_type)


def plot_samples(ax, x, y, format_str='-', c=None, label=None, alpha=0.2,
                 mode='mean_std', y_ge_0=False, std_fac=1., alpha_mean=1, 
                 y_ge=None, y_le=None, 
                 **kwargs):
    try:
        import torch
        if type(y) == torch.Tensor:
            y = y.numpy()
    except:
        "Torch not installed."
    if type(mode) == int:
        i_s = mode
        ax.plot(x, y[i_s], format_str, c=c, label=label, **kwargs)
    else: 
        if 'all' in mode:
            for i_s in range(len(y)):
                ax.plot(x, y[i_s], format_str, c=c, alpha=alpha, 
                        label=label if i_s == 0 and not 'mean' in mode else None, 
                       **kwargs)
                
        y_m = np.nanmean(y, axis=0)
        if 'mean' in mode:
            ax.plot(x, y_m, format_str, c=c, label=label, alpha=alpha_mean, **kwargs)
        if 'std' in mode or 'sd' in mode:
            y_s = np.nanstd(y, axis=0) 
            # Apply a factor (as for the standard error, but custom)
            y_s *= std_fac
            # Bound?
            if y_ge is not None:
                lower_bound = y_ge
            elif y_ge_0:
                lower_bound = 0
            else:
                lower_bound = -np.inf
            if y_le is not None:
                upper_bound = y_le
            else:
                upper_bound = np.inf
            ax.fill_between(x, np.maximum(y_m-y_s, lower_bound), np.minimum(y_m+y_s, upper_bound), 
                            color=c, alpha=alpha, linewidth=0)
        if 'se' in mode:
            y_s = np.nanstd(y, axis=0) / np.sqrt(len(y[~np.isnan(y)]) / y.shape[1])
            if y_ge_0:
                ax.fill_between(x, np.maximum(y_m-y_s, 0), y_m+y_s, color=c, alpha=alpha, linewidth=0)
            else:
                ax.fill_between(x, y_m-y_s, y_m+y_s, color=c, alpha=alpha, linewidth=0)
        # --- ADDED: median + interquartile range ---
        if 'median' in mode:
            y_med = np.nanmedian(y, axis=0)
            ax.plot(x, y_med, format_str, c=c, label=label, alpha=alpha_mean, **kwargs)
        if 'quart' in mode:
            y_med = np.nanmedian(y, axis=0)
            y_q1  = np.nanpercentile(y, 25, axis=0)
            y_q3  = np.nanpercentile(y, 75, axis=0)
            ax.fill_between(x, y_q1, y_q3, color=c, alpha=alpha, linewidth=0)
 
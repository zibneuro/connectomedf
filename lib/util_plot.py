from configparser import Interpolation
import os
import io
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import TwoSlopeNorm, SymLogNorm
#from tabulate import tabulate
from pathlib import Path
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from lib.constants import STR_SYNAPSE_COUNTS, SYNCLUSTERS_X, SYNCLUSTERS_Y, get_short_descriptor
import seaborn as sns

cmap_viridis = mpl.cm.get_cmap("viridis").copy()
cmap_viridis_reversed = mpl.cm.get_cmap("viridis").copy().reversed()
cmap_blues = mpl.cm.get_cmap("Blues").copy()
cmap_blues.set_bad(color='white')
cmap_reds = mpl.cm.get_cmap("Reds").copy()
cmap_reds.set_bad(color='white')
cmap_binary = mpl.cm.get_cmap("binary").copy()
cmap_coolwarm = mpl.cm.get_cmap("coolwarm").copy()
cmap_yellowbrown = mpl.cm.get_cmap("YlOrBr").copy()
cmap_heat = mpl.cm.get_cmap("gist_heat").copy()
cmap_heat_reversed = mpl.cm.get_cmap("gist_heat").copy().reversed()
cmap_brbg = plt.colormaps["BrBG"].copy()

# https://seaborn.pydata.org/tutorial/color_palettes.html
COLORS_CATEGORICAL = sns.color_palette("bright") 
COLORS_CATEGORICAL2 = sns.color_palette("colorblind") 
COLORS_CATEGORICAL3 = sns.color_palette("dark") 

COLOR_0 = "black"
COLOR_1 = COLORS_CATEGORICAL[0]
COLOR_2 = COLORS_CATEGORICAL[1]
COLOR_3 = COLORS_CATEGORICAL[2]
COLOR_4 = COLORS_CATEGORICAL[3]
COLOR_5 = COLORS_CATEGORICAL[4]
COLOR_6 = COLORS_CATEGORICAL[5]
COLOR_7 = COLORS_CATEGORICAL[6]
COLOR_8 = COLORS_CATEGORICAL[7]
COLOR_9 = COLORS_CATEGORICAL[8]
COLOR_10 = COLORS_CATEGORICAL[9]

COLOR_VIS_1 = COLORS_CATEGORICAL[0] # dark blue
COLOR_VIS_2 = COLORS_CATEGORICAL[9] # bright blue

COLOR_H01_1 = COLORS_CATEGORICAL[3] # dark orange
COLOR_H01_2 = COLORS_CATEGORICAL[1] # bright orange

COLOR_EXC = COLOR_9
COLOR_INH = COLOR_3
COLOR_EXC2 = COLOR_5

COLOR_EMPIRICAL = COLOR_0
COLOR_H0 = COLOR_8
COLOR_H0_dark = COLORS_CATEGORICAL3[7]
COLOR_MODEL = COLORS_CATEGORICAL3[9]

"""
COLORS_MODELS = [COLOR_EMPIRICAL, COLOR_EXTRA_3, COLOR_MODEL_1, COLOR_MODEL_2]
COLORS_EMPIRICAL_MODEL = [COLOR_MODEL_2, COLOR_MODEL_2]
COLORS_DEFAULT = [COLOR_EMPIRICAL, COLOR_EXTRA_3, "lightsteelblue", "cornflowerblue", "tab:blue"]
COLORS_INFERRED = [COLOR_EMPIRICAL, COLOR_INFERRED, COLOR_EXTRA_3]

COLORS_CATEGORICAL = [sns.color_palette("colorblind")[k] for k in [0,1,2,3,4,5,6,7,8]]
COLORS_GREYS = sns.color_palette("Greys", n_colors=12)
COLORS_GREYS_REV = list(reversed(COLORS_GREYS))
"""

class ColorInterpolator():
    def __init__(self, colormap, vmin=-1, vmax=1, scale_fn = None):
        self.colormap = colormap
        self.vmin = vmin
        self.vmax = vmax
        self.scale_fn = scale_fn

    def __call__(self, value):        
        if(self.scale_fn is not None):
            value = self.scale_fn(value)

        if(value <= self.vmin):
            t = 0
        elif(value >= self.vmax):
            t = 1
        else:  
            t = (value - self.vmin) / (self.vmax - self.vmin)
        return self.colormap(float(t))

def rgb_to_js_color(rgb_tuple):
    r = int(rgb_tuple[0] * 255)
    g = int(rgb_tuple[1] * 255)
    b = int(rgb_tuple[2] * 255)
    return f"rgb({r}, {g}, {b})"

def get_blues(n):
    colors = sns.color_palette("Blues", n)
    colors_rev = list(reversed(colors))
    return colors_rev

DISPLAY_IMAGE_TARGET_HEIGHT = 400

def figsize_mm_to_inch(width_mm, height_mm):
    conversion_factor = 0.0393701
    return (conversion_factor * width_mm, conversion_factor * height_mm)


def makeDir(dirname):
    try:
        os.mkdir(dirname)
    except:
        pass

LINEWIDTH = 0.5

# https://matplotlib.org/stable/users/explain/customizing.html#customizing
def initPlotSettings(spines_top_right = False, font_size=7, legend_font_size=7):    
    mpl.rcParams["axes.spines.top"] = spines_top_right
    mpl.rcParams["axes.spines.right"] = spines_top_right    
    mpl.rcParams["legend.frameon"] = False

    mpl.rcParams["legend.fontsize"] = legend_font_size
    mpl.rcParams["font.size"] = font_size
    mpl.rcParams["font.family"] = "Arial"    
    mpl.rcParams["axes.labelsize"] = 7
    mpl.rcParams["xtick.labelsize"] = 7
    mpl.rcParams["ytick.labelsize"] = 6
    mpl.rcParams["mathtext.default"] = "regular"

    mpl.rcParams["figure.max_open_warning"] = 0
    
    mpl.use("cairo")

def set_linewidth(ax):
    ax.spines['top'].set_linewidth(LINEWIDTH)    # Top spine
    ax.spines['right'].set_linewidth(LINEWIDTH)  # Right spine
    ax.spines['bottom'].set_linewidth(LINEWIDTH) # Bottom spine
    ax.spines['left'].set_linewidth(LINEWIDTH)   # Left spine
    ax.tick_params(axis='both', which='both', width=LINEWIDTH)


def format_labels(labels):
    formatted = []
    for label in labels:
        if("_" in label):
            formatted.append(get_short_descriptor(label))
        else:
            formatted.append(label)
    return formatted 


def getNamedColor(name):
    if(name == "empirical"):
        return "forestgreen"
    elif(name in ["rule0", "no-specificity"]):
        return "black"
    elif(name == "rule1"):
        return "dimgrey"
    elif(name == "rule2"):
        return "darkgrey"
    elif(name == "rule3"):
        return "lightgrey"
    else:
        raise ValueError(name)


def getNamedColorRBC(name):
    if(name == "rule0"):
        return "lightgrey"
    elif(name == "rule1"):
        return "lightsteelblue"
    elif(name == "rule2"):
        return "cornflowerblue"
    elif(name == "rule3"):
        return "tab:blue"
    elif(name == "inferred"):
        return "tab:red"
    elif(name == "inferred0"):
        return "lightpink"
    elif(name == "inferred1"):
        return "lightcoral"
    elif(name == "inferred2"):
        return "indianred"
    elif(name == "inferred3"):
        return "brown"
    elif(name == "inferred4"):
        return "red"
    elif(name == "empirical"):
        return "forestgreen"
    else:
        raise ValueError(name)


def get_bar_sizes(num_datasets):
    if(num_datasets == 1):
        bar_width = 0.8
        bar_offsets = [0]
    elif(num_datasets == 2):
        bar_width = 0.4
        bar_offsets = [-0.2, 0.2]
    elif(num_datasets == 3):
        bar_width = 0.25
        bar_offsets = [-0.25, 0, 0.25]
    elif(num_datasets == 4):
        bar_width = 0.2
        bar_offsets = [-0.3, -0.1, 0.1, 0.3]
    elif(num_datasets == 5):
        bar_width = 0.16
        bar_offsets = [-0.32, -0.16, 0, 0.16, 0.32]
    else:
        raise ValueError(num_datasets)
    return bar_width, bar_offsets


    


def loadObservable(filename, isScalar=True, percentileLow=25, percentileHigh=75):
    if(not os.path.exists(filename)):
        print("not found: {}".format(filename))
        exit()
    data = np.loadtxt(filename)
    if(isScalar):
        data = np.atleast_1d(data)
        mean = np.median(data)
        lower = np.abs(np.percentile(data, percentileLow) - mean)
        upper = np.abs(np.percentile(data, percentileHigh) - mean)
    else:
        data = np.atleast_2d(data)
        mean = np.median(data, axis=0)
        lower = np.abs(np.percentile(data, percentileLow, axis=0) - mean)
        upper = np.abs(np.percentile(data, percentileHigh, axis=0) - mean)
    return mean, lower, upper


def get_median_and_deviations(data, percentile_low=25, percentile_high=75):
    if(data.ndim != 2):
        raise ValueError(data.shape)
    median = np.median(data, axis=0) 
    deviations = np.zeros((2, data.shape[1]))
    deviations[0, :] = np.abs(np.percentile(data, percentile_low, axis=0) - median)
    deviations[1, :] = np.abs(np.percentile(data, percentile_high, axis=0) - median)
    return median, deviations


def get_mean_and_deviations(data, percentile_low=25, percentile_high=75):
    if(data.ndim != 2):
        raise ValueError(data.shape)
    mean = np.mean(data, axis=0)
    #std = np.std(data, axis=0)

    deviations = np.zeros((2, data.shape[1]))
    deviations[0, :] = mean - np.percentile(data, percentile_low, axis=0)
    deviations[1, :] = np.percentile(data, percentile_high, axis=0) - mean
    return mean, deviations



def loadTripletDistribution(filenameBase, filenameBaseRandom, percentileLow=25, percentileHigh=75, selectedIndices=(1, 2, 3, 4, 5, 6), returnObserved=True, returnObservedFlat=False):
    distribution_observed = np.atleast_2d(np.loadtxt(filenameBase + "_observed"))[:, selectedIndices]

    if(returnObserved and returnObservedFlat):
        raise ValueError()

    def getRatio(observed, random):
        expectedRandom = np.mean(random, axis=0)
        if(np.any(expectedRandom == 0)):
            print(expectedRandom)
            raise RuntimeError("unobserved in random network")
        ratio = np.divide(observed, expectedRandom)
        return ratio

    if(returnObserved):
        median = np.median(distribution_observed, axis=0)
        lower = np.abs(np.percentile(distribution_observed, percentileLow, axis=0) - median)
        upper = np.abs(np.percentile(distribution_observed, percentileHigh, axis=0) - median)
        return median, lower, upper
    elif(returnObservedFlat):
        return distribution_observed
    else:
        distribution_random = np.atleast_2d(np.loadtxt(filenameBaseRandom + "_random"))[:, selectedIndices]
        distribution_ratio = getRatio(distribution_observed, distribution_random)
        median = np.median(distribution_ratio, axis=0)
        lower = np.abs(np.percentile(distribution_ratio, percentileLow, axis=0) - median)
        upper = np.abs(np.percentile(distribution_ratio, percentileHigh, axis=0) - median)
        return median, lower, upper


def plot_bar_chart(datasets, colors=[COLOR_EMPIRICAL, COLOR_H0, COLOR_MODEL], x_labels=None, x_axis_label=None, y_axis_label="synapse count", y_lim=(0,5000), use_log=False, error_bars=True, dataset_labels=None, 
                   fig_size=(5, 3), filename=None, adjust_left=0.15, adjust_right=0.95, adjust_bottom=0.2, adjust_top=0.95, marker_size=1,
                   percentile_low = 25, percentile_high=75, title=None, hatch_patterns = None, hatch_linewidth=0.7):
    #mpl.rcParams["axes.spines.top"] = False
    #mpl.rcParams["axes.spines.right"] = False
    #mpl.rcParams['font.size'] = font_size
    #mpl.rcParams["legend.fontsize"] = legend_font_size
    initPlotSettings()

    num_datasets = len(datasets)
    
    # check parameter consistency
    for dataset in datasets:
        assert dataset.ndim == 2

    dimension = datasets[0].shape[1]
    for dataset_idx in range(1, num_datasets):
        assert datasets[dataset_idx].shape[1] == dimension

    if(x_labels is not None):
        assert len(x_labels) == dimension

    if(colors is not None):
        assert len(colors) == num_datasets

    if(hatch_patterns is None): 
        hatch_patterns = []
        for _ in range(0, num_datasets):
            hatch_patterns.append(None)
    else:
        assert len(hatch_patterns) == num_datasets

    bar_width, bar_offsets = get_bar_sizes(num_datasets)
    x = np.arange(0, dimension)

    # Create plot
    plt.clf()

    fig = plt.figure(figsize=fig_size)
    ax = plt.gca()
    set_linewidth(ax)


    if(use_log):
        plt.yscale("log")

    for dataset_idx in range(0, num_datasets):
        data = datasets[dataset_idx]

        if(data.shape[0] == 1):
            data_median = data[0, :]
            deviations = None
        else:
            data_median, deviations = get_median_and_deviations(data, percentile_low, percentile_high)

        if(colors is None):
            color = COLORS_CATEGORICAL[dataset_idx]
        else:
            color = colors[dataset_idx]

        if(dataset_labels is not None):
            label = dataset_labels[dataset_idx]
        else:
            label = None

        plt.bar(x+bar_offsets[dataset_idx], data_median-y_lim[0], zorder=1, bottom=y_lim[0], align='center', color=color, width=bar_width, label=label, hatch=hatch_patterns[dataset_idx])   
        if(error_bars and deviations is not None):
            plt.errorbar(x+bar_offsets[dataset_idx], data_median, zorder=2, yerr=deviations, capsize=2, capthick=LINEWIDTH, linestyle="none", ecolor="black", elinewidth=LINEWIDTH)

    plt.xlim([-0.5, max(x)+0.5])
    plt.ylim(y_lim)

    if(x_labels):
        plt.xticks(x, format_labels(x_labels))
    else:
        plt.xticks(x, x)

    if(y_axis_label):
        plt.ylabel(y_axis_label)

    if(x_axis_label):
        plt.xlabel(x_axis_label)

    if(dataset_labels):
        plt.legend()

    if(title is not None):
        plt.title(title)

    plt.subplots_adjust(left=adjust_left, right=adjust_right, bottom=adjust_bottom, top=adjust_top)

    image = savefig_png_svg(fig, filename)    
    plt.clf()
    return resize_image(image)

def plot_motifs_bar_chart(datasets, colors=[COLOR_EMPIRICAL, COLOR_H0, COLOR_MODEL], y_axis_label="occurrences\nrel. to random", y_lim=None, use_log=False, error_bars=True, dataset_labels=None,
                          fig_size=(5, 3), filename=None, adjust_left=0.15, adjust_right=0.95, adjust_bottom=0.2, adjust_top=0.95,
                          quantile_low = 25, quantile_high=75, use_mean = False, 
                          marker_size = 1, capsize=2, mew=0.5,
                          selected_motifs=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)):    
    initPlotSettings()    

    num_datasets = len(datasets)

    # check parameter consistency
    for dataset in datasets:
        assert dataset.ndim == 2

    dimension = datasets[0].shape[1]
    assert dimension == 16
    for dataset_idx in range(1, num_datasets):
        assert datasets[dataset_idx].shape[1] == dimension
    
    if(colors is not None):
        assert len(colors) == num_datasets

    x_labels = np.arange(16) + 1
    bar_width, bar_offsets = get_bar_sizes(num_datasets)

    # apply selection
    selected_indices = np.array(selected_motifs, dtype=int) - 1 
    x = np.arange(0, selected_indices.size)    
    x_labels = x_labels[selected_indices] 
    
    # Create plot
    plt.clf()
    fig, ax = plt.subplots(figsize=fig_size)
    set_linewidth(ax)

    if(use_log):
        plt.yscale("symlog")

    for dataset_idx in range(0, num_datasets):
        data = datasets[dataset_idx]

        if(data.shape[0] == 1):
            data_median = data[0, :]
            data_median = data_median[selected_indices]      
            #idx_nonzero = data_median != 0      
            #data_median[idx_nonzero] -= 1
            deviations = None
        else:
            if(use_mean):
                data_median, deviations = get_mean_and_deviations(data, quantile_low, quantile_high)
            else:
                data_median, deviations = get_median_and_deviations(data, quantile_low, quantile_high)                        
            
            data_median = data_median[selected_indices]      
            idx_nonzero = data_median != 0      
            #data_median[idx_nonzero] -= 1

            deviations = deviations[:,selected_indices]
            #deviations[:,idx_nonzero] -= 1            
        
        color = colors[dataset_idx]

        if(dataset_labels is not None):
            label = dataset_labels[dataset_idx]
        else:
            label = None

        if(use_mean):
            if(deviations is not None):
                ax.errorbar(x+bar_offsets[dataset_idx], data_median, zorder=1, yerr=deviations, markersize=marker_size, capsize=capsize, capthick=LINEWIDTH, 
                            marker="o", color=color, linestyle="none", ecolor="black", mec="black", mew=mew, elinewidth=LINEWIDTH, label=label)
            else:
                ax.plot(x+bar_offsets[dataset_idx], data_median, zorder=2, marker="D", markersize=marker_size, mec="black", mew=mew, color=color, linestyle="none", label=label)
                ax.bar(x+bar_offsets[dataset_idx], data_median-1, bottom=1, zorder=1, align='center', color=color, width=bar_width) 
        else:
            ax.bar(x+bar_offsets[dataset_idx], data_median-1, bottom=1, zorder=1, align='center', color=color, width=bar_width, label=label)
            if(error_bars and deviations is not None):
                ax.errorbar(x+bar_offsets[dataset_idx], data_median, zorder=2, yerr=deviations, capsize=2, capthick=LINEWIDTH, linestyle="none", ecolor="black", elinewidth=LINEWIDTH)

    plt.xlim([-0.5, max(x)+0.5])

    if(use_log):
        pass
        #yticks = [0,1,2,5,10,100,1000]
        #yticklabels = [0,1,2,5,10,100,1000]
        #plt.yticks(yticks, yticklabels)
    else:
        pass
        #yticks = [0,1,2,5,10,100,1000]
        #yticklabels = [0,1,2,5,10,100,1000]   
        #plt.yticks(yticks, yticklabels)


    if(y_lim is not None):
        plt.ylim(y_lim)
    else:
        plt.ylim([0,5])

    plt.xticks(x, x_labels)

    plt.xlabel("motif")
    plt.ylabel(y_axis_label)
    
    if(use_mean):
        plt.hlines(1, -0.5, max(x)+0.5, colors="grey", zorder=-1, linewidth=LINEWIDTH)
    else:
        plt.hlines(1, -0.5, max(x)+0.5, colors="black", linewidth=LINEWIDTH)

    if(dataset_labels):
        plt.legend()

    plt.subplots_adjust(left=adjust_left, right=adjust_right, bottom=adjust_bottom, top=adjust_top)
        
    image = savefig_png_svg(fig, filename)            
    plt.clf()
    return resize_image(image)


def savefig_png_svg(fig, filename, dpi = 500, save_pdf=True):
    if(filename is None):
        return
    filepath = Path(filename)        
    fig.savefig(filepath.with_suffix(".png"), dpi=dpi)    
    if(save_pdf):
        fig.savefig(filepath.with_suffix(".pdf"), transparent=True)
    img = Image.open(filepath.with_suffix(".png"))
    return img


def plot_single_motif_juxtaposed(datasets, colors, y_axis_label="occurrences rel. to random", xaxis_label=None, y_lim=None, use_log=False, error_bars=True, dataset_labels=None,
                          fig_size=(5, 3), filename=None, adjust_left=0.15, adjust_right=0.95, adjust_bottom=0.2, adjust_top=0.95, font_size=10, legend_font_size=10,
                          show_motif_symbols = False, motif_symbol_scale=1, motif_symbol_offsets=1, show_motif_ids = True, hatch_patterns = None, hatch_linewidth=0.7):
    """
    datasets = [dataset_1, dataset_2, ...]
    dataset_k = [array, array]  (e.g., empirical (array-size = 1), model (array-size = 500))
    """
    
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams['font.size'] = font_size
    mpl.rcParams['hatch.linewidth'] = hatch_linewidth

    num_datasets = len(datasets) 
    x = np.arange(num_datasets)

    # check parameter consistency
    for dataset in datasets:
        assert len(dataset) == 2 # [array1, array2]
        assert dataset[0].ndim == 1
        assert dataset[1].ndim == 1
    
    assert len(colors) == 2

    assert dataset_labels is not None
    x_labels = dataset_labels
    bar_width, bar_offsets = get_bar_sizes(2)

    # Create plot
    plt.clf()
    fig, ax = plt.subplots(figsize=fig_size)
    
    if(show_motif_symbols):
        raise NotImplementedError

    if(use_log):
        plt.yscale("symlog")


    if(hatch_patterns is None): 
        hatch_patterns = [None, None]        
    else:
        assert len(hatch_patterns) == 2

    maxY = 2

    for dataset_idx in range(0, num_datasets):

        arr1 = datasets[dataset_idx][0]
        arr2 = datasets[dataset_idx][1]

        def getPlotData(array):
            if(array.size == 1):
                data_median = array           
                idx_nonzero = data_median != 0      
                data_median[idx_nonzero] -= 1     
                deviations = None                
            else:
                data_median, deviations = get_median_and_deviations(array.reshape((-1,1)))                                                        
                idx_nonzero = data_median != 0      
                data_median[idx_nonzero] -= 1
            return data_median, deviations

        def getMaxVal(medians, deviations):
            if(deviations is None):
                return medians
            else:
                return medians + deviations[1]

        medians1, deviations1 = getPlotData(arr1)
        medians2, deviations2 = getPlotData(arr2)        
        max1 = getMaxVal(medians1, deviations1)
        max2 = getMaxVal(medians2, deviations2)
        
        maxY = max(maxY, max1+1, max2+1)

        #print(medians1, medians2)

        if(dataset_labels is not None):
            label = dataset_labels[dataset_idx]
        else:
            label = None

        def plot(data_median, deviations, idx, color, hatch):        
            ax.bar(x[dataset_idx] + bar_offsets[idx], data_median, bottom=1, zorder=1, align='center', color=color, width=bar_width, label=label, hatch=hatch)
            if(error_bars and deviations is not None):
                ax.errorbar(x[dataset_idx] + bar_offsets[idx], data_median + 1, zorder=2, yerr=deviations, capsize=2, capthick=0.7, linestyle="none", ecolor="black", elinewidth=1)

        plot(medians1, deviations1, 0, colors[0], hatch_patterns[0])
        plot(medians2, deviations2, 1, colors[1], hatch_patterns[1])

    plt.xlim([-0.5, max(x)+0.5])
    plt.xticks(x, x_labels)
    
    if(xaxis_label is not None):
        plt.xlabel(xaxis_label)
    if(y_axis_label):
        plt.ylabel(y_axis_label)

    if(use_log):        
        yticks = [0,1,2,5,10,100,1000]
        yticklabels = [0,1,2,5,10,100,1000]
        plt.yticks(yticks, yticklabels)
    else:
        pass
        #yticks, yticklabels = plt.yticks()    
        #plt.yticks(yticks-1, yticklabels)

    if(y_lim is None):        
        plt.ylim([0, maxY])
    else:
        plt.ylim(y_lim)

    plt.hlines(1, -0.5, max(x)+0.5, colors="black", linewidth=1)

    #if(dataset_labels):
    #    plt.legend(prop={'size': legend_font_size})

    plt.subplots_adjust(left=adjust_left, right=adjust_right, bottom=adjust_bottom, top=adjust_top)
    
    image = savefig_png_svg(fig, filename)    
    plt.clf()
    return image
    

def get_png_image_for_display(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    return resize_image(image)


def resize_image(image):
    width, height = image.size    
    scale_factor = DISPLAY_IMAGE_TARGET_HEIGHT / height
    new_height = int(scale_factor * height)
    new_width = int(scale_factor * width)
    return image.resize((new_width, new_height))

def resize_image_width(image, targetWidth):
    width, height = image.size    
    scale_factor = targetWidth / width
    new_height = int(scale_factor * height)
    new_width = int(scale_factor * width)
    return image.resize((new_width, new_height))


def plot_line_chart(datasets, colors=[COLOR_EMPIRICAL, COLOR_H0, COLOR_MODEL], x_labels=None, x_axis_label=SYNCLUSTERS_X, y_axis_label=SYNCLUSTERS_Y, x_lim=None, y_lim=None, use_log=False, error_bars=True, dataset_labels=None, linestyles=["x-",".--",".-"],
                    fig_size=(5, 3), filename=None, adjust_left=0.15, adjust_right=0.95, adjust_bottom=0.2, adjust_top=0.95, title=None, hline_y=None,
                    capsize=2, linewidth=LINEWIDTH,  marker_size=3):
    initPlotSettings()

    num_datasets = len(datasets)

    # check parameter consistency
    for dataset in datasets:
        assert dataset.ndim == 2

    max_dimension = 0     
    for dataset_idx in range(1, num_datasets):
        max_dimension = max(datasets[dataset_idx].shape[1], max_dimension)

    if(x_labels is not None):
        assert len(x_labels) == max_dimension

    if(colors is not None):
        assert len(colors) == num_datasets

    if(linestyles is not None):
        assert len(linestyles) == num_datasets

    if(dataset_labels is not None):
        assert len(dataset_labels) == num_datasets

    # create plot    
    plt.clf()
    fig = plt.figure(figsize=fig_size)
    ax = plt.gca()
    set_linewidth(ax)

    if(use_log):
        plt.yscale("symlog")

    for dataset_idx in range(0, num_datasets):
        data = datasets[dataset_idx]
        x = np.arange(0, data.shape[1])

        if(data.shape[0] == 1):
            data_median = data[0, :]
            deviations = None
        else:
            data_median, deviations = get_median_and_deviations(data)
        
        color = colors[dataset_idx]

        if(dataset_labels is not None):
            label = dataset_labels[dataset_idx]
        else:
            label = None

        if(linestyles is not None):
            linestyle = linestyles[dataset_idx]
        else:
            linestyle = "o-"

        if(error_bars and deviations is not None):
            plt.errorbar(x, data_median, yerr=deviations, zorder=dataset_idx, fmt=linestyle, ms=marker_size, linewidth=linewidth, capsize=capsize, elinewidth=linewidth, capthick=LINEWIDTH, c=color, label=label)
        else:
            plt.plot(x, data_median, linestyle, zorder=dataset_idx, ms=marker_size, linewidth=linewidth, c=color, label=label)

    if(x_labels):
        plt.xticks(x, x_labels)
    else:
        plt.xticks(x, x)

    if(y_lim is not None):
        plt.ylim(y_lim)

    if(x_lim is not None):
        plt.xlim(x_lim)
    else:
        plt.xlim([-0.5, max(x)+0.5])

    if(hline_y is not None):
        plt.axhline(y=hline_y, color="k", linestyle="--", linewidth=LINEWIDTH)

    if(x_axis_label):
        plt.xlabel(x_axis_label)
    if(y_axis_label):
        plt.ylabel(y_axis_label)

    if(dataset_labels):
        plt.legend()

    if(title):
        plt.title(title)

    plt.subplots_adjust(left=adjust_left, right=adjust_right, bottom=adjust_bottom, top=adjust_top)
    
    image = savefig_png_svg(fig, filename)             
    plt.clf()
    return resize_image(image)


def plot_matrix(folder_path, M, colormap, vmin=-1, vmax=1, matrix_information=None, highlight_ids=[], model_descriptor="", high_res=False):
    if(not high_res):
        plot_matrix(folder_path, M, colormap, vmin, vmax, matrix_information, highlight_ids, model_descriptor, True)

    path = Path(folder_path)
    path.mkdir(parents=True, exist_ok=True)
    assert "png" not in str(path)
    folder = Path(path.parent)/folder_path.stem
    extended_path = folder/folder_path.stem    
    high_res_path = folder/f"{folder_path.stem}_high_res"
    os.makedirs(folder, exist_ok=True)        

    initPlotSettings()
    
    mpl.rcParams["axes.spines.top"] = True
    mpl.rcParams["axes.spines.right"] = True
    mpl.rcParams['font.size'] = 5

    assert isinstance(highlight_ids, list)

    def get_minortick_locations():
        row_locations = []
        row_labels = []
        col_locations = []
        col_labels = []
        for k in range(0, len(highlight_ids)):
            neuron_id = highlight_ids[k]
            for j in range(0, len(matrix_information.ids_presynaptic)):
                if(matrix_information.ids_presynaptic[j] == neuron_id):
                    row_locations.append(j+0.5)
                    row_labels.append(k+1)
            for j in range(0, len(matrix_information.ids_postsynaptic)):
                if(matrix_information.ids_postsynaptic[j] == neuron_id):
                    col_locations.append(j+0.5)
                    col_labels.append(k+1)     
        return row_locations, row_labels, col_locations, col_labels   

    n = M.shape[0]
    m = M.shape[1]
    extent = (0, m, n, 0)

    plt.clf()
    fig = plt.figure(figsize=figsize_mm_to_inch(120,60))
    fig.subplots_adjust(left=0.05, right=0.5, bottom=0.1, top=0.9)
    ax = fig.add_subplot()
    pos = ax.imshow(M, extent=extent, cmap=colormap, vmin=vmin, vmax=vmax, interpolation='none')
    fig.colorbar(pos, ax=ax, fraction=0.05, pad=0.05)
    if(high_res):
        HIGH_RES_LINEWIDTH = 0.1
        ax.grid(color='black', linewidth=HIGH_RES_LINEWIDTH)
        ax.set_frame_on(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(HIGH_RES_LINEWIDTH)
        ax.tick_params(axis='both', which='both', width=HIGH_RES_LINEWIDTH)
    else:
        ax.grid(color='black', linewidth=0.5)
        ax.set_frame_on(True)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1)

    if(matrix_information is None):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        pos_rows, pos_cols = matrix_information.get_separator_line_positions()
        ax.set_xticks(pos_cols)
        ax.set_yticks(pos_rows)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        row_locations, row_labels, col_locations, col_labels = get_minortick_locations()        
        ax.set_xticks(col_locations, minor=True)
        ax.set_xticklabels(col_labels, minor=True)
        ax.set_yticks(row_locations, minor=True)
        ax.set_yticklabels(row_labels, minor=True)
        
        ax.tick_params(axis='both', which='minor', color="red")
        
        def write_lines(lines):
            x = 0.6
            y = 0.9
            dy = -0.04
            for line in lines:
                plt.text(x, y, line, transform=fig.transFigure, va='center')
                y += dy

        def write_title(title):
            x = 0.06
            y = 0.93
            plt.text(x, y, title, transform=fig.transFigure, va='center')
        
        def write_selections_info(selections, ids_per_selection):
            lines = []
            num_total = 0            
            for selection in selections:
                num_neurons = len(ids_per_selection[selection])
                num_total += num_neurons                
                lines.append(f"{selection} ({num_neurons})")
            return lines

        if(matrix_information.descriptor):
            if(model_descriptor):
                write_title(f"{matrix_information.descriptor} | {model_descriptor}")
            else:
                write_title(matrix_information.descriptor)

        lines = []
        lines += [f"{M.shape[0]} rows:"]
        lines += write_selections_info(matrix_information.selections_presynaptic, matrix_information.ids_presynaptic_by_selection)
        lines += [""]
        lines += [f"{M.shape[1]} columns:"]
        lines += write_selections_info(matrix_information.selections_postsynaptic, matrix_information.ids_postsynaptic_by_selection)
        lines += [""]
        lines += ["annotated neurons:"]
        for k in range(0, len(highlight_ids)):
            lines += [f"{k+1}: {highlight_ids[k]}"]

        write_lines(lines)
    
    if(high_res):
        savefig_png_svg(fig, high_res_path, dpi=2000, save_pdf=False)    
        plt.clf()
        return
    else:
        image = savefig_png_svg(fig, extended_path)             
        plt.clf()
        return resize_image(image)


def calc_deviation(x_empirical, x_model):
    assert x_empirical.shape[0] == 1

    x_empirical = x_empirical[0,:]
    x_model_median = np.median(x_model, axis=0)    

    nonzero_idx = x_empirical != 0
    x_empirical_nonzero = x_empirical[nonzero_idx]
    x_model_median_nonzero = x_model_median[nonzero_idx]

    l2 = np.linalg.norm(x_empirical - x_model_median)
    avg_pct = np.mean(np.divide(np.abs(x_empirical_nonzero - x_model_median_nonzero), x_empirical_nonzero))

    return l2, avg_pct


def print_deviations(x_empirical, x_model_1, x_model_2, label_1 = "non-specific", label_2 = "inferred"):

    def get_row(row_id, empirical, model):
        l2, avg_pct = calc_deviation(empirical, model)
        row = [row_id, "{:.2f}".format(l2), "{:.2f}".format(avg_pct)]
        return row

    table = [["model", "L2", "avg. percent"],
        get_row(label_1, x_empirical, x_model_1),
        get_row(label_2, x_empirical, x_model_2),
        ]        

    print("Deviations:")
    #print(tabulate(table, headers="firstrow", tablefmt="github"))

def get_parameter_indices(x_model, columns_percentile_bounds):
    
    def get_indices_for_column(values, prct_low, prct_high):
        assert prct_low < prct_high
        bound_low = np.percentile(values, prct_low)
        bound_high = np.percentile(values, prct_high)
        mask = (values >= bound_low) & (values <= bound_high)
        return set(np.where(mask)[0])
        
    indices = set()
    for col_idx, bounds in columns_percentile_bounds.items():                
        indices_col = get_indices_for_column(x_model[:,col_idx], *bounds)
        
        if(indices):
            indices &= indices_col
        else:
            indices = indices_col

        if(not indices):
            raise ValueError("no intersecting indices left after col_idx {}".format(col_idx))

    return sorted(indices)


def plot_probability_lines(values, labels, filename, highlight_idx_linestyle={}, line_alpha = 0.7, y_lim=None, y_label = "change in\nlikelihood",
    fig_size=(5, 3), adjust_left=0.15, adjust_right=0.95, adjust_bottom=0.2, adjust_top=0.95, use_log=False):
    
    initPlotSettings()    

    plt.clf()
    fig = plt.figure(figsize=fig_size)

    ax = plt.gca()
    set_linewidth(ax)

    x = np.arange(values.shape[1])
    for i in range(0, values.shape[0]):
        plt.plot(x, values[i,:], color="black", alpha=line_alpha, zorder=0, linewidth=LINEWIDTH)
    print(x)
    print(labels)
    plt.xticks(x, labels)
    plt.xlim(0, values.shape[1]-1)

    if(use_log):
        plt.yscale("symlog")

    for i, linestyle in highlight_idx_linestyle.items():
        plt.plot(x, values[i,:], color=linestyle[1], alpha=1, zorder=1, linestyle=linestyle[0], linewidth=1)

    if(y_lim is not None):
        plt.ylim(y_lim)
    plt.ylabel(y_label)
    
    plt.subplots_adjust(left=adjust_left, right=adjust_right, bottom=adjust_bottom, top=adjust_top)

    image = savefig_png_svg(fig, filename)    
    plt.clf()
    return resize_image(image)



def combine_images(folder=None, filenames = None, images=None, mode="horizontal", header_image=None):
    assert mode in ["horizontal", "vertical"]

    width = 0     
    height = 0

    if(images is None):
        images = []
        assert folder is not None
        assert filenames is not None

        for filename in filenames:
            image = Image.open(folder / filename)
            images.append(image)

    if(header_image is not None):
        images = [header_image] + images

    for image in images:
        w, h = image.size
        if(mode == "horizontal"):
            width += w
            height = max(height, h)
        else:
            width = max(width, w)
            height += h            

    image_combined = Image.new('RGB', (width, height))
    
    coord = 0
    for image in images:
        if(mode == "horizontal"):
            image_combined.paste(image, (coord, 0))
            coord += image.width
        else:
            image_combined.paste(image, (0, coord))
            coord += image.height

    return image_combined



class NormalizePreferenceValue:
    def __init__(self, min_value, max_value, mid_value = 1):
        self.min_value = min_value
        self.mid_value = mid_value
        self.max_value = max_value

    def __call__(self, M):
        M_clipped = np.clip(M, a_min=self.min_value, a_max=self.max_value)
        norm_fn = TwoSlopeNorm(vmin=self.min_value, vcenter=self.mid_value, vmax=self.max_value) 
        return 2 * norm_fn(M_clipped) - 1
    
class NormalizeDeviationValue:
    def __init__(self, min_value, max_value, mid_value = 0):
        self.min_value = min_value
        self.mid_value = mid_value
        self.max_value = max_value

    def __call__(self, M):
        M_clipped = np.clip(M, a_min=self.min_value, a_max=self.max_value)
        norm_fn = TwoSlopeNorm(vmin=self.min_value, vcenter=self.mid_value, vmax=self.max_value) 
        return norm_fn(M_clipped)
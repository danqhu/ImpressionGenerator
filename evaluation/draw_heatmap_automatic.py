import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import matplotlib as mpl

import pandas as pd


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, vmin=0, vmax=0.5, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.6, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    # cbar.set_ticks(ticks=[1,2,3,4,5], labels=['Very Bad', 'Bad', 'Fair', 'Good', 'Very Good'])
    

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, left=False, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    
    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    a = im.norm(data)

    
    threshold_up = 0.75

    threshold_down = 0.25

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
        

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            kw.update(color=textcolors[int((im.norm(data[i, j]) < threshold_down) or (im.norm(data[i, j]) > threshold_up))])
            val = valfmt(data[i, j], None)
            if val == '--': val = ''
            text = im.axes.text(j, i, val, **kw)
            texts.append(text)

    return texts




metrics = [
    "Tongyi Qianwen (CT)",
    "ERNIE Bot (CT)", 
    "ChatGPT (CT)", 
    "Bard (CT)", 
    "Baichuan (CT)", 
    "ChatGLM (CT)", 
    "HuatuoGPT (CT)", 
    "ChatGLM_Med (CT)",
    "",
    "Tongyi Qianwen (PET-CT)",
    "ERNIE Bot (PET-CT)", 
    "ChatGPT (PET-CT)", 
    "Bard (PET-CT)", 
    "Baichuan (PET-CT)", 
    "ChatGLM (PET-CT)", 
    "HuatuoGPT (PET-CT)", 
    "ChatGLM_Med (PET-CT)",
    "",
    "Tongyi Qianwen (US)",
    "ERNIE Bot (US)", 
    "ChatGPT (US)", 
    "Bard (US)", 
    "Baichuan (US)", 
    "ChatGLM (US)", 
    "HuatuoGPT (US)", 
    "ChatGLM_Med (US)",
    ]
llms = [
    "BLEU1 (zero-shot)", 
    "BLEU2 (zero-shot)", 
    "BLEU3 (zero-shot)", 
    "BLEU4 (zero-shot)", 
    "ROUGE-L (zero-shot)", 
    "METEOR (zero-shot)",
    "",
    "BLEU1 (one-shot)", 
    "BLEU2 (one-shot)", 
    "BLEU3 (one-shot)", 
    "BLEU4 (one-shot)", 
    "ROUGE-L (one-shot)", 
    "METEOR (one-shot)",
    "",
    "BLEU1 (three-shot)", 
    "BLEU2 (three-shot)", 
    "BLEU3 (three-shot)", 
    "BLEU4 (three-shot)", 
    "ROUGE-L (three-shot)", 
    "METEOR (three-shot)"
    ]

file_name = 'All_raw_new'

values = pd.read_excel("automatic_eval_fig\{}.xlsx".format(file_name))

values = np.float64(values.to_numpy()[:,1:])




fig, ax = plt.subplots()


im, cbar = heatmap(values, metrics, llms, ax=ax,
                cmap="RdBu", cbarlabel="")   # YlGn
texts = annotate_heatmap(im, valfmt="{x:.3f}")

fig.tight_layout()
fig.set_size_inches(20,20)
# plt.show()



fig.savefig('automatic_eval_fig\{}.pdf'.format(file_name),dpi=600,format='pdf',bbox_inches='tight')


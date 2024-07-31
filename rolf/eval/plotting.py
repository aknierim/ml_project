import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter
from sklearn.metrics import confusion_matrix

LABELS_MAP = {
    0: "FRI",
    1: "FRII",
    2: "Compact",
    3: "Bent",
}


def plot_train_val_rocauc(train, val, ax=None):
    if ax is None:
        ax = plt.gca()

    ax.plot(train["Step"], train["Value"], label="Train ROC AUC")
    ax.plot(val["Step"], val["Value"], label="Validation ROC AUC")

    ax.set(xlabel="Step", ylabel="Accuracy")

    ax.legend()


def plot_sample(images, labels, labels_pred, axs=None):
    if axs is None:
        axs = plt.gca()

    for ax, img, label, label_pred in zip(axs, images, labels, labels_pred):
        img = img.squeeze()
        label = label.item()
        label_pred = label_pred

        correct = label_pred == label

        ax.text(
            0.05,
            0.95,
            f"Truth: {LABELS_MAP[label]}",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
            color="white",
            fontsize=16,
        )
        ax.text(
            0.05,
            0.85,
            "Pred:",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
            color="white",
            fontsize=16,
        )
        ax.text(
            0.295,
            0.85,
            f"{LABELS_MAP[label_pred]}",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
            color="limegreen" if correct else "red",
            fontsize=16,
        )
        ax.patch.set_edgecolor("limegreen" if correct else "red")
        ax.patch.set_linewidth(5)

        ax.set(
            xticks=[],
            xticklabels=[],
            yticks=[],
            yticklabels=[],
        )
        ax.imshow(img, cmap="inferno")

    return ax


def conf_matrix(
    y_truth, y_pred, sample_weight=None, normalize=None, valfmt="{x:1.0f}", **kwargs
):
    cm = confusion_matrix(
        y_truth, y_pred, sample_weight=sample_weight, normalize=normalize
    )

    im, cbar = plot_confusion_matrix(cm, valfmt=valfmt, **kwargs)

    return im, cbar, cm


def plot_confusion_matrix(
    data,
    labels=None,
    ax=None,
    cbar_kw=None,
    cbarlabel="",
    annotate=True,
    valfmt="{x:1.0f}",
    **kwargs,
) -> tuple:
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data : array_like
        A 2D numpy array of shape (N, N).
    labels : array_like, optional
        A list or array of length N with the labels.
    ax : matplotlib.axes.Axes, optional
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.
    cbar_kw : dict, optional
        A dictionary with arguments to `matplotlib.Figure.colorbar`.
    cbarlabel : str, optional
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.

    Returns
    -------
    im : imshow object
        Imshow object.
    cbar : cbar object
        Colorbar object.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    im = ax.imshow(data, **kwargs)

    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.set_xticks(
        np.arange(data.shape[1]),
        labels,
        rotation=-30,
        ha="left",
        rotation_mode="anchor",
    )
    ax.set_yticks(np.arange(data.shape[0]), labels=labels)
    ax.set(xlabel="Predicted Labels", ylabel="True Labels")

    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    if annotate:
        _annotate_heatmap(im, valfmt=valfmt)

    return im, cbar


def _annotate_heatmap(
    im,
    data=None,
    valfmt="{x:1.0f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw,
):
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

    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    if isinstance(valfmt, str):
        valfmt = StrMethodFormatter(valfmt)

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

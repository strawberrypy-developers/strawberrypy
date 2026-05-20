import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..classes import Model


def plot_marker(
    model: Model,
    marker: np.ndarray,
    savefig: bool = False,
    filename: str = "marker.pdf",
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    r"""
    Plot the local topological marker in real space. This function is intended for periodic
    boundary conditions (PBC) only since within open boundary conditions the trace over the
    whole sample vanishes by construction.

    Parameters
    ----------
        model : Model
            The model for which to plot the topological marker.
        marker : np.ndarray
            The PBC local topological marker to be plotted.
        title : str
            Title of the plot. Default is :python:`'Local Chern marker'`.
        savefig : bool
            If :python:`True`, saves the figure in a file with name given by :python:`filename`.
            Default is :python:`False`.
        filename : str
            Name of the file where to save the figure if :python:`savefig == True`. Default is
            :python:`'local_marker.pdf'`.
        **kwargs
            Additional keyword arguments to pass to the scatter plot.

    Returns
    -------
        fig : plt.Figure
            The figure object.
        ax : plt.Axes
            The axes object.
    """
    # Only the master rank in parallel can plot and save
    if not model.backend.is_master_rank:
        return None, None

    fig, ax = plt.subplots(1, 1)

    # Extract the x and y coordinates of the cartesian positions
    x, y = model.cart_positions[:, 0], model.cart_positions[:, 1]

    # Default style if no additional keyword arguments are provided
    if kwargs.get("cmap", None) is None:
        kwargs["cmap"] = "viridis"
    if kwargs.get("s", None) is None:
        kwargs["s"] = 10
    if kwargs.get("edgecolors", None) is None:
        kwargs["edgecolors"] = "none"
    if kwargs.get("vmin", None) is None:
        kwargs["vmin"] = np.min(marker)
    if kwargs.get("vmax", None) is None:
        kwargs["vmax"] = np.max(marker)

    # Plot the marker as a scatter plot
    sc = ax.scatter(x, y, c=marker, **kwargs)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    _ = fig.colorbar(sc, cax=cax, label="Marker")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    if savefig:
        fig.savefig(filename, bbox_inches="tight")

    return fig, ax

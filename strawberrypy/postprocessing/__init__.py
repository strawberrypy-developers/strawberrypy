r"""
A module for post-processing utilities to perform averages and constractions in real space,
compute single-point invariants from local topological markers, plot markers in real space
and more.
"""

from .tracemarker import trace_marker
from .plotting import plot_marker
from .pair_distance import get_pbc_distance_pairs
from .contractions import lattice_contraction, pbc_lattice_contraction
from .averages import average_over_radius
from .marker_io import save_marker, load_marker
from .correlations import correlation

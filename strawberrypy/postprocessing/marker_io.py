import os
import numpy as np


def save_marker(
    directory: str, filename: str, marker: np.ndarray, header: dict = {}
) -> None:
    r"""
    Save marker data to a compressed ``.npz`` file. The marker data is saved along with
    an optional header containing metadata. The header is stored as a dictionary, and the
    marker data is stored as a NumPy array.

    Parameters
    ----------
        directory : str
            The directory where the marker data will be saved. If the directory does not
            exist, it will be created.
        filename : str
            The name of the file (without extension) to save the marker data.
        marker : np.ndarray
            The marker data to be saved.
        header : dict, optional
            A dictionary containing header information to be saved with the marker data.
    """
    # Remove the .npz extension if user forgot to exclude it
    if filename.endswith(".npz"):
        filename = filename[:-4]

    # Ensure the path is correctly formed
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f"{filename}.npz")

    np.savez_compressed(filepath, header=header, data=marker)

    print(f"Marker data saved to: {filepath}", flush=True)


def load_marker(directory: str, filename: str) -> tuple[dict, np.ndarray]:
    r"""
    Load marker data from a compressed ``.npz`` file. The function returns a tuple
    containing the loaded header (:python:`dict`) and marker data (:python:`np.ndarray`).

    Parameters
    ----------
        directory : str
            The directory where the marker data is saved.
        filename : str
            The name of the file (without extension) containing the marker data.

    Returns
    -------
        tuple[dict, np.ndarray]
            A tuple containing the loaded header (:python:`dict`) and marker data
            (:python:`np.ndarray`).
    """
    # Remove the .npz extension if provided
    if filename.endswith(".npz"):
        filename = filename[:-4]

    # Ensure the path is correctly formed
    filepath = os.path.join(directory, f"{filename}.npz")

    if not os.path.exists(f"{filepath}"):
        raise FileNotFoundError(f"File {filepath} not found.")

    npz_loaded = np.load(f"{filepath}", allow_pickle=True)
    return (npz_loaded["header"], npz_loaded["data"])

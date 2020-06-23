import numpy as np

def rms(amplitudes, reduction=np):
    """
    Compute the root mean square amplitude of a field stored on a uniform grid.

    Parameters
    ----------
    amplitudes : :obj:`np.array`
        Array containing the amplitudes of the field
    reduction: default: numpy module
        class or module implementing reduction operation sum
    Returns
    -------
    rms_height : float
        Root mean square height value.
    """
    n = np.prod(amplitudes.shape)
    # if topography.is_MPI:

    return np.sqrt(
        reduction.sum((amplitudes - pnp.sum(amplitudes) / n) ** 2) / n)

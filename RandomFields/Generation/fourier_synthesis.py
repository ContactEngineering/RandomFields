import numpy as np


def _irfft23(karr, rarr):
    """
    wrapper around irfft functions for convenience in calling them

    Parameters
    ----------
    karr : array_like
        Fourier-space representation, shape: (nx//2 + 1) or (nx//2 + 1, ny) or
        (nx//2 + 1, ny, nz)

    rarr : array_like
        Real-space representation, shape: (nx) or (nx, ny) or (nx, ny, nz)
    """
    qarr = np.squeeze(karr)
    dim = qarr.ndim
    if dim == 2:
        _irfft2(karr, rarr)
    elif dim == 3:
        _irfft3(karr, rarr)
    else:
        raise ValueError(
            'The needed irfft is not implemented for dim={}'.format(dim))


def _irfft2(karr, rarr):
    """
    Inverse 2d real-to-real FFT

    Parameters
    ----------
    karr : array_like
        Fourier-space representation, shape: nx//2 + 1, ny

    rarr : array_like
        Real-space representation, shape: nx, ny

    """
    rx, ry = rarr.shape
    arr = karr.copy()
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    nx, ny = arr.shape

    # This makes sure that no extraneous elements of the first row is utilized
    arr[0, :] = np.fft.irfft(arr[0, 0:ny // 2 + 1], n=ny)

    if rx % 2 == 0:
        """
        This makes sure that no extraneous elements of the last row is utilized
        in case of having even number of rows
        """
        arr[-1, :] = np.fft.irfft(arr[-1, 0:ny // 2 + 1], n=ny)
    else:
        arr[-1, :] = np.fft.ifft(arr[-1, :])

    # ifft is applied on the rest of the rows
    for i in np.arange(1, nx - 1):
        arr[i, :] = np.fft.ifft(arr[i, :])

    # irfft applied in the x direction
    for j in np.arange(ny):
        rarr[:, j] = np.fft.irfft(arr[:, j], n=rx)
    # rarr = np.fft.irfft(arr, axis=1, n=rx)


def _irfft3(karr, rarr):
    """
    Inverse 3d real-to-real FFT

    Parameters
    ----------
    karr : array_like
        Fourier-space representation, shape: nx//2 + 1, ny, nz

    rarr : array_like
        Real-space representation, shape: nx, ny, nz

    """
    nx, ny, nz = karr.shape
    rx, ry, rz = rarr.shape
    arr = np.zeros_like(karr)

    # This makes sure no extraneous elements of the first plane is utilized
    _irfft2(karr[0, 0:ny // 2 + 1, :], arr[0, :, :])

    if rx % 2 == 0:
        """
        This makes sure no extraneous elements of the last plane is utilized
        in case of having even number of rows
        """
        _irfft2(karr[-1, 0:ny // 2 + 1, :], arr[-1, :, :])
    else:
        arr[-1, :, :] = np.fft.ifft2(karr[-1, :, :])

    # ifft2 is applied on the rest of the planes
    for i in np.arange(1, nx - 1):
        arr[i, :, :] = np.fft.ifft2(karr[i, :, :])

    # irfft applied in the x direction
    for j in np.arange(ny):
        for k in np.arange(nz):
            rarr[:, j, k] = np.fft.irfft(arr[:, j, k], n=rx)
    # rarr = np.fft.irfft(arr, axis=0, n=rx)


def self_affine_prefactor(dim, nb_grid_pts, physical_sizes, Hurst,
                          rms_height=None, rms_slope=None,
                          short_cutoff=None, long_cutoff=None):
    r"""
    Compute prefactor :math:`C_0` for the power-spectrum density of an ideal
    self-affine topography given by

    .. math ::

        C(q) = C_0 q^{-2-2H}

    for two-dimensional topography maps and

    .. math ::

        C(q) = C_0 q^{-1-2H}

    for one-dimensional line scans. Here :math:`H` is the Hurst exponent.

    Note:
    In the 2D case:

    .. math ::

        h^2_{rms} = \frac{1}{2 \pi} \int_{0}^{\infty} q C^{iso}(q) dq

    whereas in the 1D case:

    .. math ::

        h^2_{rms} = \frac{1}{\pi} \int_{0}^{\infty} C^{1D}(q) dq

    See Equations (1) and (4) in [1].


    Parameters
    ----------
    nb_grid_pts : array_like
        Resolution of the topography map or the line scan.
    physical_sizes : array_like
        Physical physical_sizes of the topography map or the line scan.
    Hurst : float
        Hurst exponent.
    rms_height : float
        Root mean-squared height.
    rms_slope : float
        Root mean-squared slope of the topography map or the line scan.
    short_cutoff : float
        Short-wavelength cutoff.
    long_cutoff : float
        Long-wavelength cutoff.

    Returns
    -------
    prefactor : float
        Prefactor :math:`\sqrt{C_0}`

    References
    -----------
    [1]: Jacobs, Junge, Pastewka, Surf. Topgogr.:
         Metrol. Prop. 5, 013001 (2017)

    """

    nb_grid_pts = np.asarray(nb_grid_pts)
    physical_sizes = np.asarray(physical_sizes)

    if short_cutoff is not None:
        q_max = 2 * np.pi / short_cutoff
    else:
        q_max = np.pi * np.min(nb_grid_pts / physical_sizes)

    if long_cutoff is not None:
        q_min = 2 * np.pi / long_cutoff
    else:
        q_min = 2 * np.pi * np.max(1 / physical_sizes)

    area = np.prod(physical_sizes)

    if rms_height is not None:
        # Assuming no rolloff region
        fac = (2 * rms_height /
               np.sqrt(q_min ** (-2 * Hurst) -
                       q_max ** (-2 * Hurst)) * np.sqrt(Hurst * np.pi))
    elif rms_slope is not None:
        fac = 2 * rms_slope / np.sqrt(
            q_max ** (2 - 2 * Hurst) -
            q_min ** (2 - 2 * Hurst)) * np.sqrt((1 - Hurst) * np.pi)
    else:
        raise ValueError('Neither rms height nor rms slope is defined!')

    if dim == 1:
        fac /= np.sqrt(2)

    return fac * np.prod(nb_grid_pts) / np.sqrt(area)


def fourier_synthesis(nb_grid_pts, physical_sizes, hurst,
                      rms_height=None, rms_slope=None, c0=None,
                      short_cutoff=None, long_cutoff=None, rolloff=1.0,
                      amplitude_distribution=lambda n: np.random.normal(
                          size=n),
                      phases_maker=lambda m: np.exp(2 * np.pi *
                                                    np.random.rand(m) * 1j),
                      rfn=None, kfn=None):
    """
    Create a self-affine, randomly rough surface using a Fourier filtering
    algorithm. The algorithm is described in:
    Ramisetti et al., J. Phys.: Condens. Matter 23, 215004 (2011);
    Jacobs, Junge, Pastewka, Surf. Topgogr.: Metrol. Prop. 5, 013001 (2017)

    Parameters
    ----------
    nb_grid_pts : array_like
        Resolution of the field.
    physical_sizes : array_like
        Physical sizes of the periodic box.
    hurst : float
        Hurst exponent.
    rms_height : float
        Root mean-squared height.
    rms_slope : float
        Root mean-squared slope.
    c0: float
        self affine prefactor :math:`C_0`:
        :math:`C(q) = C_0 q^{-2-2H}`
    short_cutoff : float
        Short-wavelength cutoff.
    long_cutoff : float
        Long-wavelength cutoff.
    rolloff : float
        Value for the power-spectral density (PSD) below the long-wavelength
        cutoff. This multiplies the value at the cutoff, i.e. unit will give a
        PSD that is flat below the cutoff, zero will give a PSD that is
        vanishes below cutoff. (Default: 1.0)
    amplitude_distribution : function
        Function that generates the distribution of amplitudes.
        (Default: np.random.normal)
    rfn : str
        Name of file that stores the real-space array. If specified, real-space
        array will be created as a memory mapped file. This is useful for
        creating very large topography maps. (Default: None)
    kfn : str
        Name of file that stores the Fourire-space array. If specified,
        real-space array will be created as a memory mapped file.
        This is useful for creating very large topography maps. (Default: None)
    progress_callback : function(i, n)
        Function that is called to report progress.

    Returns
    -------
    array: np.array
        random field values
    """
    dim = len(nb_grid_pts)
    max_dim = 3
    if short_cutoff is not None:
        q_max = 2 * np.pi / short_cutoff
    else:
        q_max = np.pi * np.min(np.asarray(nb_grid_pts)
                               / np.asarray(physical_sizes))

    if long_cutoff is not None:
        q_min = 2 * np.pi / long_cutoff
    else:
        q_min = None

    if c0 is None:
        fac = self_affine_prefactor(dim, nb_grid_pts, physical_sizes,
                                    hurst, rms_height=rms_height,
                                    rms_slope=rms_slope,
                                    short_cutoff=short_cutoff,
                                    long_cutoff=long_cutoff)
    else:
        # prefactor for the fourier heights
        # C(q) = c0 q^(-2-2H) = 1 / A |fh(q)|^2
        # and h(x,y) = sum(1/A fh(q) e^(iqx)))
        #                    ▼ compensate for the np.fft normalisation
        fac = np.sqrt(c0) * np.prod(nb_grid_pts) / \
              np.sqrt(np.prod(physical_sizes))

    n = np.ones(max_dim, dtype=int)
    s = np.ones(max_dim)
    n[0:dim:1] = nb_grid_pts
    s[0:dim:1] = physical_sizes
    # kshape: the shape of the fourier series coeffs considering
    # the symmetry of real Fourier transform
    kshape = n
    kn = n[0] // 2 + 1  # SYMMETRY
    kshape[0] = kn

    rarr = np.empty(nb_grid_pts, dtype=np.float64)
    karr = np.empty(kshape, dtype=np.complex128)

    qx = 2 * np.pi * np.arange(kn) / s[0]
    for z in range(n[2]):
        if z > n[2] // 2:
            qz = 2 * np.pi * (n[2] - z) / s[2]
        else:
            qz = 2 * np.pi * z / s[2]

        for y in range(n[1]):
            if y > n[1] // 2:
                qy = 2 * np.pi * (n[1] - y) / s[1]
            else:
                qy = 2 * np.pi * y / s[1]
            q_sq = qz ** 2 + qy ** 2 + qx ** 2
            if z == 0 and y == 0:
                q_sq[0] = 1.
            # making phases and amplitudes of the wave funcrion with
            # random generating functions
            # this functions could be passed to the function in the
            # first place and you can see their default
            # functions in the signature of the function
            phase = phases_maker(kn)
            ran = fac * phase * amplitude_distribution(kn)
            karr[:, y, z] = ran * q_sq ** (-((dim * 0.5) + hurst) / 2)
            karr[q_sq > q_max ** 2, y, z] = 0.
            if q_min is not None:
                mask = q_sq < q_min ** 2
                karr[mask, y, z] = (rolloff * ran[mask] *
                                    q_min ** (-((dim * 0.5) + hurst)))

    if dim == 1:
        rarr = np.fft.irfft(karr.T)
    else:
        _irfft23(karr, rarr)
    return rarr

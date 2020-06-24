from RandomFields.Generation import fourier_synthesis

import pytest
import numpy as np
from RandomFields.Analysis.scalar_parameters import rms


def test_fourier_synthesis_rms_height_more_wavevectors():
    """
    Set amplitude to 0 (rolloff = 0) outside the self affine region.

    Long cutoff wavelength is smaller then the box size so that we get closer
    to a continuum of wavevectors
    """
    n = 256
    H = 0.74
    rms_height = 7.
    s = 1.

    realised_rms_heights = []
    for i in range(100):
        topography = fourier_synthesis((n, n), (s, s),
                                       H,
                                       rms_height=rms_height,
                                       rolloff=0,
                                       long_cutoff=s / 8,
                                       short_cutoff=4 * s / n,
                                       # amplitude_distribution=lambda n: np.ones(n)
                                       )

        realised_rms_heights.append(rms(topography))
    # print(abs(np.mean(realised_rms_heights) - rms_height) / rms_height)
    assert abs(np.mean(realised_rms_heights) - rms_height) / \
           rms_height < 0.1  # TODO: this is not very accurate !


def test_fourier_synthesis_rms_height():
    n = 256
    H = 0.74
    rms_height = 7.
    s = 1.

    realised_rms_heights = []
    for i in range(100):
        topography = fourier_synthesis((n, n), (s, s),
                                       H,
                                       rms_height=rms_height,
                                       long_cutoff=None,
                                       short_cutoff=4 * s / n,
                                       # amplitude_distribution=lambda n: np.ones(n)
                                       )
        realised_rms_heights.append(rms(topography))
    assert abs(np.mean(realised_rms_heights) - rms_height) / \
           rms_height < 0.3  # TODO: this is not very accurate !


def test_fourier_synthesis_1D_input():
    H = 0.7
    c0 = 1.

    n = 512
    s = n * 4.
    ls = 8
    qs = 2 * np.pi / ls
    np.random.seed(0)
    topography = fourier_synthesis((n,), (s,),
                                   H,
                                   c0=c0,
                                   long_cutoff=s / 2,
                                   short_cutoff=ls,
                                   amplitude_distribution=lambda n: np.ones(n)
                                   )


def test_fourier_synthesis_linescan_hrms_more_wavevectors():
    """
    Set amplitude to 0 (rolloff = 0) outside the self affine region.

    Long cutoff wavelength is smaller then the box size so that we get closer
    to a continuum of wavevectors
    """
    H = 0.7
    hrms = 4.
    n = 4096
    s = n * 4.
    ls = 8
    qs = 2 * np.pi / ls
    np.random.seed(0)
    realised_rms_heights = []
    for i in range(50):
        t = fourier_synthesis((n,), (s,),
                              rms_height=hrms,
                              hurst=H,
                              rolloff=0,
                              long_cutoff=s / 8,
                              short_cutoff=ls,
                              )
        realised_rms_heights.append(rms(t))
    realised_rms_heights = np.array(realised_rms_heights)
    ref_height = hrms
    # print(np.sqrt(np.mean((realised_rms_heights - np.mean(realised_rms_heights))**2)))
    assert abs(np.mean(realised_rms_heights) -
               ref_height) / ref_height < 0.1  #

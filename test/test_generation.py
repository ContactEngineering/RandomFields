#
# Copyright 2020 Antoine Sanner
#
# ### MIT license
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

from RandomFields.Generation import fourier_synthesis

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
    np.random.seed(0)
    realised_rms_heights = []
    for i in range(100):
        topography = fourier_synthesis((n, n), (s, s),
                                       H,
                                       rms_height=rms_height,
                                       rolloff=0,
                                       long_cutoff=s / 8,
                                       short_cutoff=4 * s / n,
                                       # amplitude_distribution=lambda n: np.ones(n) # noqa E501
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
    np.random.seed(0)
    realised_rms_heights = []
    for i in range(100):
        topography = fourier_synthesis((n, n), (s, s),
                                       H,
                                       rms_height=rms_height,
                                       long_cutoff=None,
                                       short_cutoff=4 * s / n,
                                       # amplitude_distribution=lambda n: np.ones(n) # noqa E501
                                       )
        realised_rms_heights.append(rms(topography))
    assert abs(np.mean(realised_rms_heights) - rms_height) / \
        rms_height < 0.3  # TODO: this is not very accurate !


def test_fourier_synthesis_1d_input():
    H = 0.7
    c0 = 1.

    n = 512
    s = n * 4.
    ls = 8
    np.random.seed(0)
    fourier_synthesis((n,), (s,),
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
    # print(np.sqrt(np.mean((realised_rms_heights
    #                         - np.mean(realised_rms_heights))**2)))
    assert abs(np.mean(realised_rms_heights) -
               ref_height) / ref_height < 0.1  #


def test_fourier_synthesis_3D():
    """
    Testing the 3D random field generated via fourier synthesis
    """
    n = [101, 101, 101]
    nx, ny, nz = n
    s = [101, 101, 101]
    sx, sy, sz = s
    g = [sx/nx, sy/ny, sz/nz]
    gx, gy, gz = g
    hurst = 0.8
    rms_height = 0.5

    Topography = fourier_synthesis((nx, ny, nz), (sx, sy, sz),
                                   hurst, rms_height=rms_height,
                                   long_cutoff=sx / 3.)

    rmss = np.zeros(nx)
    for i in range(nx):
        topo = Topography[i, :, :]
        rmss[i] = rms(topo)

    mean_rms_x = np.mean(rmss)
    assert(((mean_rms_x - rms_height) / rms_height) < 1e-5)

    rmss = np.zeros(ny)
    for i in range(ny):
        topo = Topography[:, i, :]
        rmss[i] = rms(topo)

    mean_rms_y = np.mean(rmss)
    assert(((mean_rms_y - rms_height) / rms_height) < 1e-5)

    rmss = np.zeros(nz)
    for i in range(nz):
        topo = Topography[:, :, i]
        rmss[i] = rms(topo)

    mean_rms_z = np.mean(rmss)
    assert(((mean_rms_z - rms_height) / rms_height) < 1e-5)

    rms_3d = rms(Topography)
    assert(((rms_3d - rms_height) / rms_height) < 1e-5)

from .Random_Field_fourier_synthesis import fourier_synthesis

from .PyCo.SurfaceAnalysis import CharacterisePeriodicSurface

from .PyCo.UniformLineScanAndTopography import Topography, UniformLineScan
# from .common import compute_wavevectors, ifftn

import numpy as np
import scipy.stats as stats

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import pytest
import numpy as np
from NuMPI import MPI


def test_fourier_synthesis_3D():
    n = [101, 101, 101]
    nx, ny, nz = n
    s = [101, 101, 101]
    sx, sy, sz = s
    g = [sx/nx, sy/ny, sz/nz]
    gx, gy, gz = g
    hurst = 0.8
    rms_height = 0.5

    x = np.arange(0, sx, gx)
    y = np.arange(0, sy, gy)
    z = np.arange(0, sz, gz)
    X, Y = np.meshgrid(x, y)

    c = fourier_synthesis((nx, ny, nz), (sx, sy, sz),
                          hurst, rms_height=rms_height,
                          long_cutoff=sx / 3.)

    hursts = np.zeros(nx)
    rmss = np.zeros(nx)
    for i in range(nx):
        physical_sizes = np.array([sy, sz])
        topo = Topography(c[i, :, :], physical_sizes, True)
        analysis = CharacterisePeriodicSurface(topo)
        rmss[i] = np.sqrt(np.mean(c[i, :, :] ** 2))
        hursts[i] = analysis.estimate_hurst()

    mean_hurst_x = np.mean(hursts)
    mean_rms_x = np.mean(rmss)
    assert(((mean_hurst_x - hurst) / hurst) < 0.5)
    assert(((mean_rms_x - rms_height) / rms_height) < 0.5)

    hursts = np.zeros(ny)
    rmss = np.zeros(ny)
    for i in range(ny):
        physical_sizes = np.array([sx, sz])
        topo = Topography(c[:, i, :], physical_sizes, True)
        analysis = CharacterisePeriodicSurface(topo)
        rmss[i] = np.sqrt(np.mean(c[:, i, :] ** 2))
        hursts[i] = analysis.estimate_hurst()
    mean_hurst_y = np.mean(hursts)
    mean_rms_y = np.mean(rmss)
    assert(((mean_hurst_y - hurst) / hurst) < 0.5)
    assert(((mean_rms_y - rms_height) / rms_height) < 0.5)

    hursts = np.zeros(nz)
    rmss = np.zeros(nz)
    for i in range(nz):
        physical_sizes = np.array([sx, sy])
        topo = Topography(c[:, :, i], physical_sizes, True)
        analysis = CharacterisePeriodicSurface(topo)
        rmss[i] = np.sqrt(np.mean(c[:, :, i] ** 2))
        hursts[i] = analysis.estimate_hurst()

    mean_hurst_z = np.mean(hursts)
    mean_rms_z = np.mean(rmss)
    assert(((mean_hurst_z - hurst) / hurst) < 1e-5)
    assert(((mean_rms_z - rms_height) / rms_height) < 1e-5)

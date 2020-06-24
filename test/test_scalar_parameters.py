from RandomFields.Analysis.scalar_parameters import rms
import numpy as np


def test_rms_sinewave2D():
    n = 256
    X, Y = np.mgrid[slice(0, n), slice(0, n)]

    hm = 0.1
    L = float(n)
    sinsurf = np.sin(2 * np.pi / L * X) * np.sin(2 * np.pi / L * Y) * hm

    numerical = rms(sinsurf)
    analytical = np.sqrt(hm ** 2 / 4)

    assert numerical == analytical

"""

tests if irfft is already imposing the hermitian-symmetries the way we expect

If the spectrum should be real, we expect irfft to discard the imaginary part

If the spectrum is the complex conjugate of another part of

numpy/fft/fftpack.py
numpy's irfftn
```
    # The copy may be required for multithreading.
    a = array(a, copy=True, dtype=complex)
    s, axes = _cook_nd_args(a, s, axes, invreal=1)
    for ii in range(len(axes)-1):
        a = ifft(a, s[ii], axes[ii], norm)
    a = irfft(a, s[-1], axes[-1], norm)
    return a
```

# this doesn't look good...
"""
import pytest
import numpy as np
from RandomFields.Generation.fourier_synthesis import _irfft2, _irfft3


def test_rfft_deletes_imag_part_where_spectrum_should_be_real():
    original_data = np.random.uniform(size=4)

    # of course if we
    spectrum = np.fft.rfft(original_data)
    spectrum[1] = spectrum[0] + 4j
    back = np.fft.irfft(spectrum)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(original_data, back)

    spectrum = np.fft.rfft(original_data)
    spectrum[0] = spectrum[0] + 4j
    back = np.fft.irfft(spectrum)
    np.testing.assert_allclose(original_data, back)

    # on even number of points the highest wavevector should also be real
    spectrum = np.fft.rfft(original_data)
    spectrum[-1] = spectrum[-1] + 4j
    back = np.fft.irfft(spectrum)
    np.testing.assert_allclose(original_data, back)


def test_ifft_realpart_is_not_rfft():
    """
    rfft doesn't use the negative part of the spectrum

    if the negative q part of the spectrum is not the complex conjugate of the
    positive q part, the ifft will be complex, and the realpart of the ifft will
    also be different to the rfft

    """
    n = 8
    original_data = np.random.uniform(size=n)

    # inside of a irfft2, the q=0 line that should have a real back transform.
    # does the ifft yield the correct real part also if the symmetry is broken ?

    spectrum = np.fft.fft(original_data)
    # we hope that the negative frequencies are considered as the
    # spectrum[-2] = np.random.uniform() + 1j * np.random.uniform()
    back_irfft = np.fft.irfft(spectrum[:n // 2 + 1], n=n)
    back_ifft = np.fft.ifft(spectrum).real
    np.testing.assert_allclose(back_ifft.real, back_irfft)

    with pytest.raises(AssertionError):
        spectrum = np.fft.fft(original_data)
        # we hope that the negative frequencies are considered as the
        spectrum[-2] = np.random.uniform() + 1j * np.random.uniform()
        back_irfft = np.fft.irfft(spectrum[:n // 2 + 1], n=n)
        back_ifft = np.fft.ifft(spectrum).real
        np.testing.assert_allclose(back_ifft.real, back_irfft)


# %% Test our fft

@pytest.mark.parametrize("nx", (3, 4, 8, 9))
@pytest.mark.parametrize("ny", (3, 4, 8, 9))
def test_cycle2d(nx, ny):
    real_space_original = np.random.normal(size=(nx, ny))

    spectrum = np.fft.rfft2(real_space_original.transpose()).transpose()
    print(spectrum.shape)

    real_space_back_transform = np.zeros_like(real_space_original)
    _irfft2(spectrum, real_space_back_transform)

    np.testing.assert_allclose(real_space_back_transform, real_space_original)


@pytest.mark.parametrize("nx", (3, 4, 8, 9))
@pytest.mark.parametrize("ny", (3, 4, 8, 9))
@pytest.mark.parametrize("nz", (3, 4, 8, 9))
def test_cycle3d(nx, ny, nz):
    real_space_original = np.random.normal(size=(nx, ny, nz))

    spectrum = np.fft.rfftn(real_space_original.transpose()).transpose()
    print(spectrum.shape)

    real_space_back_transform = np.zeros_like(real_space_original)
    _irfft3(spectrum, real_space_back_transform)

    np.testing.assert_allclose(real_space_back_transform, real_space_original)


def test_minimal_spectrum_unchanged_2D():
    """
    I call minimal_spectrum the part that contains no duplicates due
    to hermitian symmetry

    We request that in our implementation q[0, qy<0] is discarded amd only
    q[0, qy>0] is kept.
    """

    nx = 3
    ny = 3

    karr = np.random.normal(size=(nx // 2 + 1, ny))
    biased_karr = karr.copy()
    biased_karr[0, -1] = np.random.normal()  # changing this value shouldn't affect the result

    real = np.zeros((nx, ny))
    real_biased = np.zeros((nx, ny))

    _irfft2(biased_karr, real_biased)
    _irfft2(karr, real)

    np.testing.assert_allclose(real_biased, real)

    nx = 4
    ny = 3

    karr = np.random.normal(size=(nx // 2 + 1, ny))
    biased_karr = karr.copy()
    biased_karr[0, -1] = np.random.normal()  # changing this value shouldn't affect the result
    biased_karr[-1, -1] = np.random.normal()  # changing this value shouldn't affect the result

    real = np.zeros((nx, ny))
    real_biased = np.zeros((nx, ny))

    _irfft2(biased_karr, real_biased)
    _irfft2(karr, real)

    np.testing.assert_allclose(real_biased, real)

    # just to state this is not the case with numpy
    with pytest.raises(AssertionError):
        nx = 4
        ny = 3

        karr = np.random.normal(size=(nx // 2 + 1, ny))
        biased_karr = karr.copy()
        biased_karr[0, -1] = np.random.normal()  # changing this value
        #                                          shouldn't affect the result
        biased_karr[-1, -1] = np.random.normal()  # changing this value
        #                                           shouldn't affect the result

        real = np.zeros((nx, ny))
        real_biased = np.zeros((nx, ny))

        real_biased = np.fft.irfft2(biased_karr.T).T
        real = np.fft.irfft2(karr.T).T

        np.testing.assert_allclose(real_biased, real)


def test_minimal_spectrum_unchanged_3D():
    nx = 3
    ny = 3
    nz = 3

    karr = np.random.normal(size=(nx // 2 + 1, ny, nz))
    biased_karr = karr.copy()
    biased_karr[0, -1, :] = np.random.normal(size=nz)  # changing this value shouldn't affect the result

    real = np.zeros((nx, ny, nz))
    real_biased = np.zeros((nx, ny, nz))

    _irfft3(biased_karr, real_biased)
    _irfft3(karr, real)

    np.testing.assert_allclose(real_biased, real)

    nx = 4
    ny = 3
    nz = 3

    karr = np.random.normal(size=(nx // 2 + 1, ny, nz))
    biased_karr = karr.copy()
    biased_karr[0, -1, :] = np.random.normal(size=nz)  # changing this value shouldn't affect the result
    biased_karr[-1, -1, :] = np.random.normal(size=nz)  # changing this value shouldn't affect the result

    real = np.zeros((nx, ny, nz))
    real_biased = np.zeros((nx, ny, nz))

    _irfft3(biased_karr, real_biased)
    _irfft3(karr, real)

    np.testing.assert_allclose(real_biased, real)

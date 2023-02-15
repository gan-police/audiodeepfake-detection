"""PyTorch compatible cwt code.

Based on https://github.com/PyWavelets/pywt/blob/master/pywt/_cwt.py
Port of ptwt.continuous_transform.
"""
from typing import Tuple, Union

import numpy as np
import torch
from ptwt.continuous_transform import _integrate_wavelet, _next_fast_len
from pywt import ContinuousWavelet, DiscreteContinuousWavelet, Wavelet
from torch.fft import fft, ifft


def cwt(
    data: torch.Tensor,
    scales: Union[np.ndarray, torch.Tensor],  # type: ignore
    wavelet: Union[ContinuousWavelet, str],
) -> torch.Tensor:  # type: ignore
    """Compute the single dimensional continuous wavelet transform.

    This function is a PyTorch port of pywt.cwt as found at:
    https://github.com/PyWavelets/pywt/blob/master/pywt/_cwt.py

    Args:
        data (torch.Tensor): The input tensor of shape [batch_size, time].
        scales (torch.Tensor or np.array):
            The wavelet scales to use. One can use
            ``f = pywt.scale2frequency(wavelet, scale)/sampling_period`` to determine
            what physical frequency, ``f``. Here, ``f`` is in hertz when the
            ``sampling_period`` is given in seconds.
        wavelet (ContinuousWavelet or str): The continuous wavelet to work with.

    Raises:
        ValueError: If a scale is too small for the input signal.

    Returns:
        out_tensor (torch.Tensor): A tensor with the transformation matrix.

    Example:
        >>> import torch, ptwt
        >>> import numpy as np
        >>> import scipy.signal as signal
        >>> t = np.linspace(-2, 2, 800, endpoint=False)
        >>> sig = signal.chirp(t, f0=1, f1=12, t1=2, method="linear")
        >>> widths = np.arange(1, 31)
        >>> cwtmatr, freqs = ptwt.cwt(
        >>>     torch.from_numpy(sig), widths, "mexh", sampling_period=(4 / 800) * np.pi
        >>> )
    """
    # accept array_like input; make a copy to ensure a contiguous array
    if not isinstance(
        wavelet, (ContinuousWavelet, Wavelet, _DifferentiableContinuousWavelet)
    ):
        wavelet = DiscreteContinuousWavelet(wavelet)
    if type(scales) is torch.Tensor:
        scales = scales.cpu().numpy()
    if np.isscalar(scales):
        scales = np.array([scales])

    if isinstance(wavelet, torch.nn.Module):
        if data.is_cuda:
            wavelet.cuda()

    precision = 10
    int_psi, x = _integrate_wavelet(wavelet, precision=precision)
    if type(wavelet) is ContinuousWavelet:
        int_psi = np.conj(int_psi) if wavelet.complex_cwt else int_psi
        int_psi = torch.tensor(int_psi, device=data.device)
    elif isinstance(wavelet, torch.nn.Module):
        int_psi = torch.conj(int_psi) if wavelet.complex_cwt else int_psi
    else:
        int_psi = torch.tensor(int_psi, device=data.device)
        x = torch.tensor(x, device=data.device)

    size_scale0 = -1
    fft_data = None

    out = []
    for scale in scales:
        step = x[1] - x[0]
        j = torch.arange(
            (scale * (x[-1] - x[0]) + 1), device=data.device, dtype=data.dtype
        ) / (scale * step)
        j = torch.floor(j).type(torch.long)
        if j[-1] >= len(int_psi):
            # j = np.extract(j < len(int_psi), j)
            j = torch.masked_select(j, j < len(int_psi))
        int_psi_scale = int_psi[j].flip(0)

        # The padding is selected for:
        # - optimal FFT complexity
        # - to be larger than the two signals length to avoid circular
        #   convolution
        size_scale = _next_fast_len(data.shape[-1] + len(int_psi_scale) - 1)
        if size_scale != size_scale0:
            # Must recompute fft_data when the padding size changes.
            fft_data = fft(data, size_scale, dim=-1)
        size_scale0 = size_scale
        fft_wav = fft(int_psi_scale, size_scale, dim=-1)
        conv = ifft(fft_wav * fft_data, dim=-1)
        conv = conv[..., : data.shape[-1] + len(int_psi_scale) - 1]

        coef = -np.sqrt(scale) * torch.diff(conv, dim=-1)

        # transform axis is always -1
        d = (coef.shape[-1] - data.shape[-1]) / 2.0
        if d > 0:
            coef = coef[..., int(np.floor(d)) : -int(np.ceil(d))]
        elif d < 0:
            raise ValueError("Selected scale of {} too small.".format(scale))

        out.append(coef)
    out_tensor = torch.stack(out)
    if type(wavelet) is Wavelet:
        out_tensor = out_tensor.real
    elif isinstance(wavelet, torch.nn.Module):
        out_tensor = out_tensor if wavelet.complex_cwt else out_tensor.real
    else:
        out_tensor = out_tensor if wavelet.complex_cwt else out_tensor.real

    if isinstance(wavelet, torch.nn.Module):
        if data.is_cuda:
            wavelet.cuda()

    return out_tensor


def _frequency2scale(
    wavelet: Union[ContinuousWavelet, str],  # type: ignore
    freq: torch.Tensor,
) -> torch.Tensor:
    """Convert from to normalized frequency to CWT "scale".

    Args:
        wavelet: Wavelet instance or str
            Wavelet to integrate.  If a string, should be the name of a wavelet.
        freq (torch.Tensor):
            Frequency, normalized so that the sampling frequency corresponds to a
            value of 1.0.

    Raises:
        ValueError: If wavelet is not differentiable.

    Returns:
        scale (torch.Tensor): Corresponding scale tensor.

    """
    if not isinstance(wavelet, _DifferentiableContinuousWavelet):
        raise ValueError(
            "Wavelet must be differentiable. \
            Otherwise use pywt.frequency2scale instead."
        )

    return wavelet.center / freq


class _WaveletParameter(torch.nn.Parameter):
    pass


class _DifferentiableContinuousWavelet(
    torch.nn.Module, ContinuousWavelet  # type: ignore
):
    """A base class for learnable Continuous Wavelets."""

    def __init__(self, name: str, requires_grad: bool = True):
        """Create a trainable shannon wavelet."""
        super().__init__()
        super(ContinuousWavelet, self).__init__()

        self.dtype = torch.float64
        # Use torch nn parameter
        self.bandwidth_par = _WaveletParameter(
            torch.sqrt(torch.tensor(self.bandwidth_frequency, dtype=self.dtype)),
            requires_grad=requires_grad,
        )
        self.center_par = _WaveletParameter(
            torch.sqrt(torch.tensor(self.center_frequency, dtype=self.dtype)),
            requires_grad=requires_grad,
        )

    @property
    def bandwidth(self) -> torch.Tensor:
        """Square the bandwith parameter to ensure positive values."""
        return self.bandwidth_par * self.bandwidth_par

    @property
    def center(self) -> torch.Tensor:
        """Square the bandwith parameter to ensure positive values."""
        return self.center_par * self.center_par

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return numerical values for the wavelet on a grid."""
        return self.wavefun(10)

    def wavefun(
        self, precision: int, dtype: torch.dtype = torch.float64
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Define a grid and evaluate the wavelet on it."""
        length = 2**precision
        # load the bounds from untyped pywt code.
        lower_bound: float = float(self.lower_bound)  # type: ignore
        upper_bound: float = float(self.upper_bound)  # type: ignore
        grid = torch.linspace(
            lower_bound,
            upper_bound,
            length,
            dtype=dtype,
            device=self.bandwidth_par.device,
        )
        return self(grid), grid


class _ShannonWavelet(_DifferentiableContinuousWavelet):
    """A differentiable Shannon wavelet."""

    def __call__(self, grid_values: torch.Tensor) -> torch.Tensor:
        """Return numerical values for the wavelet on a grid."""
        shannon = (
            torch.sqrt(self.bandwidth)
            * (
                torch.sin(torch.pi * self.bandwidth * grid_values)
                / (torch.pi * self.bandwidth * grid_values)
            )
            * torch.exp(1j * 2 * torch.pi * self.center * grid_values)
        )
        return shannon


class _ComplexMorletWavelet(_DifferentiableContinuousWavelet):
    """A differentiable Shannon wavelet."""

    def __call__(self, grid_values: torch.Tensor) -> torch.Tensor:
        """Return numerical values for the wavelet on a grid."""
        morlet = (
            1.0
            / torch.sqrt(torch.pi * self.bandwidth)
            * torch.exp(-(grid_values**2) / self.bandwidth)
            * torch.exp(1j * 2 * torch.pi * self.center * grid_values)
        )
        return morlet


def get_diff_wavelet(
    wavelet: str,
) -> _ComplexMorletWavelet | _ShannonWavelet:
    """Get differentiable wavelet from given string.

    Raises:
        ValueError: If wavelet is not a str or pywt Continuous Wavelet.
        NotImplementedError: If requested wavelet is not implemented yet.
    """
    if not isinstance(wavelet, (str, ContinuousWavelet)):
        raise ValueError(
            "Wavelet must be String or pywt.continuous_transform.ContinuousWavelet."
        )
    if isinstance(wavelet, ContinuousWavelet):
        wavelet = wavelet.name

    if "cmor" in wavelet:
        return _ComplexMorletWavelet(name=wavelet)
    elif "shan" in wavelet:
        return _ShannonWavelet(name=wavelet)
    else:
        raise NotImplementedError

import numpy as np
import torch
from scipy.io import loadmat
from scipy.linalg import dft
from scipy.stats import iqr


def get_cir(datapath, dist_bounds=None):
    """
    Function that loads a CIR .mat file into a numpy array

    Parameters
    ----------
    datapath: str or datapath
        The path to the CIR file to be loaded

    dist_bounds: tuple or list or numpy array
        Initial and final indices indicating the start and end range bins to select

    Returns
    ----------
    cir_data: numpy array
        CIR data with shape (n_range_bins, n_packets, n_bp)

    """
    f = loadmat(datapath)
    complex_cir_data = f["FRAMES"]["CIR"][0][0]
    if dist_bounds:
        cir_data = complex_cir_data[:, :, dist_bounds[0] : dist_bounds[1]]
    else:
        cir_data = complex_cir_data
    cir_data = np.transpose(cir_data, (2, 1, 0))
    return cir_data


def get_subj_from_filename(filename):
    return filename.split("_")[1]


def complex_to_real_vector(complex_vec):
    """Convert a complex-valued K-dimensional vector h_c into real-valued 2K-dimensional vector h_r, s.t.:
    h_r = [Re(h_c) -Im(h_c)]

    Input: Flattened, 1-dimensional complex-valued array of shape (..., n,);
    Output: Flattened, 1-dimensional real-valued array of shape (..., 2n).
    """

    if not (
        complex_vec.dtype == np.dtype("complex128")
        or complex_vec.dtype == np.dtype("complex64")
    ):
        raise Exception("Input 'complex_vec' is not complex.")

    real_vec = np.concatenate([complex_vec.real, -complex_vec.imag], -1)
    return real_vec

def complex_to_real_vector1(complex_vec):
    """My version to obtain a real vector of shape (2,K) where the first channel 
    corresponds to the real part, and the second channel corresponds to the imag part.
    """

    if not (
        complex_vec.dtype == np.dtype("complex128")
        or complex_vec.dtype == np.dtype("complex64")
    ):
        raise Exception("Input 'complex_vec' is not complex.")

    real_vec = torch.empty((2, complex_vec.shape[0], complex_vec.shape[1], complex_vec.shape[2]))
    real_vec[0, :, :, :] = torch.Tensor(complex_vec.real)
    real_vec[1, :, :, :] = torch.Tensor(complex_vec.imag)
    return real_vec

def moving_iqr(X, moving=True, alpha=0.1, k=8):
    # compute moving standard deviation of differentiated signal
    X_diff = X[:, 1:] - X[:, :-1]
    # add a column of zeros to the left of the signal
    X_diff = torch.cat((torch.zeros(X.shape[0], 1), X_diff), dim=1)

    if moving == False:
        out = torch.tensor(iqr(X_diff, axis=1))
        return out
    else:
        out = []
        for i in range(X.shape[1]):
            # compute std of last k samples of differentiated signal and compute ema
            if i > k:
                diff_std = (
                    alpha * iqr(X_diff[:, i - k : i + 1], axis=1)
                    + (1 - alpha) * diff_std
                )
            else:
                diff_std = iqr(X_diff[:, : i + 2], axis=1)

            out.append(torch.tensor(diff_std))

        # return spikes, moving_std, moving_threshold
        return torch.stack(out, axis=1)


def process_cpx_crop(complex_crop):
    """Input: single complex crop of shape (110, NWIN)"""
    # 3.1) Take np.abs(crop) ** 2
    p = np.abs(complex_crop) ** 2

    # 3.2) Sum along range axis
    mD = p.sum(0)
    # 3.3) Make mD Shift
    mD_shift = np.roll(mD, mD.shape[0] // 2)
    return mD_shift


def min_max_freq(spec, eps=1e-8):
    """
    Min-max normalization for microDoppler spectrograms in the frequency domain

    Parameters
    ----------
    spec: numpy array, shape (n_time_frames, ndoppler)
        The spectrogram to be normalized

    eps: float (optional)
        Small positive number to avoid division by 0

    Returns
    ----------
    spec: numpy array
        normalized spectrogram with values in [0, 1], shape (n_time_frames, ndoppler)
    """

    return (spec - spec.min(1, keepdims=True)) / (
        spec.max(1, keepdims=True) - spec.min(1, keepdims=True) + eps
    )


def mD_spectrum_(complex_cir, nwin, step, n_kept_bins):

    chunks = []
    mD_columns = []
    full_spec = []

    for i in range(0, complex_cir.shape[1] - nwin, step):
        chunk = complex_cir[:, i : i + nwin]

        # ==== IHT on full X ===
        full_mask = np.ones(nwin)
        keep_idx = np.argwhere(full_mask).squeeze()
        full_chunk = chunk[:, keep_idx]
        win = np.hanning(chunk.shape[1]).reshape(1, -1)
        full_win = win[:, keep_idx]
        psi = partial_fourier(nwin, keep_idx)
        rep_psi = np.tile(psi, (complex_cir.shape[0], 1, 1))
        full_spectrum = iht(
            rep_psi, full_chunk * full_win, fixed_iters=False, n_iters=0
        )

        # Tracking on the range bin, by selecting the top k rows with
        # the highest moving interquartile range

        real_chunk = torch.tensor(complex_to_real_vector(chunk))

        iqrs_r = moving_iqr(
            real_chunk[:, : real_chunk.shape[1] // 2], k=16, moving=False
        )
        iqrs_i = moving_iqr(
            real_chunk[:, real_chunk.shape[1] // 2 :], k=16, moving=False
        )
        iqrs = torch.norm(torch.stack((iqrs_r, iqrs_i), axis=0), dim=0)
        topn_idx = torch.topk(iqrs, n_kept_bins, largest=True, sorted=True)[1]

        chunks.append(chunk[topn_idx, :])

        full_spec.append(full_spectrum[topn_idx, :].squeeze())

        mD_shift = process_cpx_crop(full_spectrum[topn_idx, :].squeeze())
        mD = min_max_freq(mD_shift[np.newaxis, :])
        mD_columns.append(mD.squeeze())

    chunks = np.array(chunks)
    full_spec = np.array(full_spec)
    mD_columns = np.array(mD_columns)

    return chunks, full_spec, mD_columns


def partial_fourier(N, idxs):
    F = np.conj(dft(N, scale="sqrtn"))
    F_part = F[idxs]  # * np.sqrt(N / len(self.keep_idxs))
    return F_part.squeeze()


def hard_thresholding(vector, sparsity_level):
    tozero = np.argpartition(np.abs(vector), -sparsity_level, axis=1)[
        :, : vector.shape[1] - sparsity_level, :
    ]
    rows = np.repeat(np.arange(len(tozero)), tozero.shape[1])
    cols = tozero.reshape(-1)
    vector[rows, cols] = complex(0, 0)
    return vector


def iht(psi, y, fixed_iters, n_iters, s=5, mu=1, maxit=300, change_conv=1e-2):
    it = 0
    end = False
    N = psi.shape[2]
    z_old = np.zeros((y.shape[0], N, 1), dtype=np.complex128)
    z = np.zeros((y.shape[0], N, 1), dtype=np.complex128)
    y = y[..., np.newaxis]
    residuals = []
    while not end:
        it += 1
        z += mu * np.transpose(np.conj(psi), (0, 2, 1)) @ (y - psi @ z)
        z = hard_thresholding(z, sparsity_level=s)
        change = np.linalg.norm(z - z_old, axis=1)
        z_old = np.copy(z)
        end_maxit = it >= maxit
        residuals.append(np.linalg.norm(psi @ z - y, axis=1).max())
        converge_thr = change_conv * np.linalg.norm(z_old, axis=1)
        # end_converged_change = change < change_conv * np.linalg.norm(z_old)
        end_conv_change = np.all(change < converge_thr)
        if fixed_iters:
            end = it == n_iters
        else:
            end = end_maxit or end_conv_change
    # print(end_maxit, it)
    # plt.plot(residuals)
    # plt.show()
    return z

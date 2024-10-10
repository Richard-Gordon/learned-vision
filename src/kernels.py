import numpy as np
from PIL import Image


def kernels_to_rgb_grid(kernels: list[np.ndarray], normalize=True) -> np.ndarray:
    """ Create a grid of RGB representations of the kernels """
    kernels_grid = np.array([kernel_to_rgb(k, normalize) for k in kernels])
    # Switch form horizontal to vertical
    kernels_grid = np.moveaxis(kernels_grid, 0, 2)
    # Flatten kernels into a grid
    kernels_grid = np.reshape(kernels_grid, (9,-1,3))
    return kernels_grid


def kernel_to_rgb(kernel: np.ndarray, normalize=True) -> np.ndarray:
    """ Convert a single kernel to a triplet of RGB representations """
    assert kernel.shape[0]==3, 'Only 3-channel kernels supported'

    # Normalize to range -1 to +1, optionally
    if normalize:
        kernel_max = np.abs(kernel).max()
        if kernel_max > 0:
            kernel /= kernel_max

    # Move the RGB channels to the end, as standard for images
    kernel = np.moveaxis(kernel.copy(), 0, -1)

    # Split into full-range, positive, and negative parts, range now 0 to 1
    kernels = (
        (kernel + 1) / 2,
        np.where(kernel<0, 0,  kernel),
        np.where(kernel>0, 0, -kernel),
    )

    # Map to unsigned 8-bit integers
    kernels = np.array([float_to_uint8(k) for k in kernels])
    return kernels


def float_to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x * 255), 0, 255).astype(np.uint8)

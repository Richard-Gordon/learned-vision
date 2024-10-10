import umap
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import kernels
import numpy as np


def umap_fit_transform(data, **kwargs):
    """ Instantiate UMAP and fit data """
    mapper = umap.UMAP(**kwargs)
    embedding_coords = mapper.fit_transform(data)
    y_coords, x_coords = embedding_coords[...,0], embedding_coords[...,1]
    return x_coords, y_coords


def plot_kernels(X, Y, weights, range='full'):
    """ Scatterplot kernels with RGB representation """
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('grey')
    fig.set_size_inches(10, 10)

    range_sel = {'full': 0, 'pos': 1, 'neg': 2}[range]

    for i, k in enumerate(weights):
        x,y = X[i], Y[i]
        kernel_rgb = kernels.kernel_to_rgb(k).astype(np.float64)/255
        kernel_rgb = kernel_rgb[range_sel]
        kernel_rgb = np.repeat(kernel_rgb, 3, axis=0)
        kernel_rgb = np.repeat(kernel_rgb, 3, axis=1)
        kernel_rgb = add_border(kernel_rgb)

        im = OffsetImage(kernel_rgb, zoom=1, interpolation='Nearest', dpi_cor=False)
        ab = AnnotationBbox(im, (x,y), frameon=False)
        ax.add_artist(ab)

    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.axis('off')


def add_border(data, size=1, color=[0,0,0]):
    """ Add a colored border to an rgb array """
    assert data.ndim == 3 and data.shape[2] == 3
    bordered_shape = (
        data.shape[0] + 2*size,
        data.shape[1] + 2*size,
        3
    )
    bordered = np.full(bordered_shape, color, data.dtype)
    bordered[1:-1, 1:-1] = data
    return bordered

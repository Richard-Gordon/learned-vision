import umap
import matplotlib.pyplot as plt

def umap_fit_transform(data, **kwargs):
    mapper = umap.UMAP(**kwargs)
    embedding_coords = mapper.fit_transform(data)
    y_coords, x_coords = embedding_coords[...,0], embedding_coords[...,1]
    return y_coords, x_coords


def plot_embeddings(x, y, colours=None):
    fig, ax = plt.subplots()
    fig.patch.set_facecolor('grey')
    fig.set_size_inches(10, 10)
    ax.scatter(x, y, c=colours)
    ax.axis('off')
    plt.show()

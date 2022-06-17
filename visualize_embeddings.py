import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA

def visualize_embeddings(pfc_embeddings, vtx_embeddings, pfc_truth, save_path):
    # given the embeddings of pfc and vtx, perform PCA and plot the embeddings
    # in 2D space
    # represent particles with dots and vertices with stars
    # save the plot to save_path
    '''
    pfc_embeddings: (N, embedding_dim)
    vtx_embeddings: (m, embedding_dim)
    pfc_truth: (N)
    save_path: string
    return: None
    '''
    pca = PCA(n_components=2)
    # transform both vertices and particles to 2D space
    # concatenate the embeddings of vertices and particles
    embeddings = np.concatenate((pfc_embeddings, vtx_embeddings), axis=0)
    embeddings_2d = pca.fit_transform(embeddings)
    # separate the embeddings of particles and vertices
    pfc_embeddings_2d = embeddings_2d[:pfc_embeddings.shape[0]]
    vtx_embeddings_2d = embeddings_2d[pfc_embeddings.shape[0]:]
    # plot the embeddings
    fig, ax = plt.subplots()
    # plot the particles
    ax.scatter(pfc_embeddings_2d[:, 0], pfc_embeddings_2d[:, 1], c=pfc_truth, cmap=cm.get_cmap('jet', 10))
    # plot the vertices
    # the color of vertices is index
    ax.scatter(vtx_embeddings_2d[:, 0], vtx_embeddings_2d[:, 1], c=np.arange(vtx_embeddings_2d.shape[0]), cmap=cm.get_cmap('jet', 10))



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA

from models.model import Net
import torch
from upuppi_v0_dataset import UPuppiV0
from torch_geometric.data import DataLoader
# random seed
np.random.seed(0)
torch.manual_seed(0)

def visualize_embeddings(pfc_embeddings, vtx_embeddings, pfc_truth, vtx_truth, save_path):
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
    
    plt.scatter(pfc_embeddings_2d[:, 0], pfc_embeddings_2d[:, 1], c=pfc_truth, cmap=cm.get_cmap('jet'))
    cbar = plt.colorbar()
    # plot the vertices
    plt.scatter(vtx_embeddings_2d[:, 0], vtx_embeddings_2d[:, 1], c=vtx_truth, cmap=cm.get_cmap('jet'), marker='*', s=100)
    # the color of vertices is index
    # add colorbar
    # save the plot
    plt.savefig('/work/submit/cfalor/upuppi/deepjet-geometric/results/{}'.format(save_path))
    plt.close()


def distinguish_neutral_charged_embeddings(pfc_embeddings, pfc_truth, save_path, x_pfc):
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
    # embeddings = np.concatenate((pfc_embeddings, vtx_embeddings), axis=0)
    embeddings_2d = pca.fit_transform(pfc_embeddings)
    # embeddings_2d = pca.fit_transform(embeddings)
    # separate the embeddings of particles and vertices
    neutral_idx = torch.nonzero(x_pfc[:,11] == 0).squeeze()
    charged_idx = torch.nonzero(x_pfc[:,11] != 1).squeeze()
    neutral_embeddings_2d = embeddings_2d[neutral_idx]
    charged_embeddings_2d = embeddings_2d[charged_idx]
    print(neutral_embeddings_2d.shape)
    print(charged_embeddings_2d.shape)
    # plot the embeddings
    # vtx_embeddings_2d = embeddings_2d[pfc_embeddings.shape[0]:]
    # plot the embeddings
    fig, ax = plt.subplots()
    # plot the particle
    # blue for neutral, red for charged
    ax.scatter(charged_embeddings_2d[:, 0], charged_embeddings_2d[:, 1], c='red', marker='.', s=5)
    ax.scatter(neutral_embeddings_2d[:, 0], neutral_embeddings_2d[:, 1], c='blue', marker='.', s=0.5)
    
    # plot the vertices
    # the color of vertices is index
    # ax.scatter(vtx_embeddings_2d[:, 0], vtx_embeddings_2d[:, 1], c=np.arange(vtx_embeddings_2d.shape[0]), cmap=cm.get_cmap('jet', 10), marker='*', s=100)
    # save the plot
    plt.savefig('/work/submit/cfalor/upuppi/deepjet-geometric/results/{}'.format(save_path))
    plt.close()

if __name__ == '__main__':
    # test visualize_embeddings
    # load the model
    data_test = UPuppiV0("/work/submit/cfalor/upuppi/deepjet-geometric/test/")
    model = "embedding_model"
    model = "GravNetConv"
    model = "combined_model"
    # model = "combined_model2"
    test_loader = DataLoader(data_test, batch_size=1, shuffle=True, follow_batch=['x_pfc', 'x_vtx'])
    model_dir = '/work/submit/cfalor/upuppi/deepjet-geometric/models/{}/'.format(model)

    # load the model
    epoch_num = 9
    upuppi_state_dict = torch.load(model_dir + 'epoch-{}.pt'.format(epoch_num))['model']
    net = Net()
    net.load_state_dict(upuppi_state_dict)
    net.eval()
    with torch.no_grad():
        data = next(iter(test_loader))
        # x_pfc = data.x_pfc.view(1, -1)
        # x_vtx = data.x_vtx.view(1, -1)
        pfc_truth = data.y.detach().numpy()
        vtx_truth = data.x_vtx[:, 2].detach().numpy()
        out, batch, pfc_embeddings, vtx_embeddings = net(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
        # visualize the embeddings
        visualize_embeddings(pfc_embeddings.cpu().numpy(), vtx_embeddings.cpu().numpy(), pfc_truth, vtx_truth, '{}_embeddings.png'.format(model))
        # distinguish neutral and charged embeddings
        distinguish_neutral_charged_embeddings(pfc_embeddings.cpu().numpy(), pfc_truth, 'vis_nc_emb{}.png'.format(epoch_num), data.x_pfc)


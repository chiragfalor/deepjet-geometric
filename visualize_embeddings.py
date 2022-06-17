
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA

from models.model2 import Net
import torch
from upuppi_v0_dataset import UPuppiV0
from torch_geometric.data import DataLoader

# random seed
np.random.seed(0)
torch.manual_seed(23)

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


if __name__ == '__main__':
    # test visualize_embeddings
    # load the model
    data_test = UPuppiV0("/work/submit/cfalor/upuppi/deepjet-geometric/test/")
    model = "embedding_model"
    test_loader = DataLoader(data_test, batch_size=1, shuffle=True, follow_batch=['x_pfc', 'x_vtx'])
    model_dir = '/work/submit/cfalor/upuppi/deepjet-geometric/models/{}/'.format(model)

    # load the model
    epoch_num = 0
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
        pfc_embeddings, vtx_embeddings = net(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
        # visualize the embeddings
        
        neutral_idx = torch.nonzero(data.x_pfc[:,11] == 0).squeeze()
        charged_idx = torch.nonzero(data.x_pfc[:,11] != 0).squeeze()
        neutral_embeddings = pfc_embeddings[neutral_idx]
        neutral_truth = pfc_truth[neutral_idx]
        visualize_embeddings(neutral_embeddings.cpu().numpy(), vtx_embeddings.cpu().numpy(), neutral_truth, vtx_truth, 'vis_emb{}.png'.format(epoch_num))


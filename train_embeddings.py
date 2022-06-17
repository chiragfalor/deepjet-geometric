import time
import sklearn
import numpy as np
import sys
#sys.path.append('/home/yfeng/UltimatePuppi/deepjet-geometric/')
from upuppi_v0_dataset import UPuppiV0
from torch_geometric.data import DataLoader
import os
import torch
from torch import nn
from models.model2 import Net
from tqdm import tqdm


BATCHSIZE = 32
start_time = time.time()
print("Training...")
# data_train = UPuppiV0("/work/submit/cfalor/upuppi/z_reg/train/")
data_test = UPuppiV0("/work/submit/cfalor/upuppi/z_reg/test/")


train_loader = DataLoader(data_test, batch_size=BATCHSIZE, shuffle=True,
                          follow_batch=['x_pfc', 'x_vtx'])
test_loader = DataLoader(data_test, batch_size=BATCHSIZE, shuffle=True,
                         follow_batch=['x_pfc', 'x_vtx'])

model = "embedding_model"
model_dir = '/work/submit/cfalor/upuppi/deepjet-geometric/models/{}/'.format(model)
#model_dir = '/home/yfeng/UltimatePuppi/deepjet-geometric/models/v0/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
# print the device used
print("Using device: ", device, torch.cuda.get_device_name(0))

# create the model
hidden_dim = 32
upuppi = Net(hidden_dim).to(device)
optimizer = torch.optim.Adam(upuppi.parameters(), lr=0.001)


def train():
    upuppi.train()
    euclidean_loss = nn.MSELoss().to(device)
    counter = 0
    total_loss = 0
    for data in tqdm(train_loader):
        counter += 1
        data = data.to(device)
        optimizer.zero_grad()
        pfc_enc, vtx_enc = upuppi(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
        # want to optimize loss, which embeds pfc with same truth closer and vtx furthest away
        # pfc loss is the distance between the embedding of the pfc and the embedding of the true vertex
        # split pfc_enc into batches
        # pfc_enc has shape (batch_size*number of particles, hidden_dim)
        # x_pfc_batch has shape (batch_size*number of particles) storing the corresponding batch index
        loss = 0
        total_pfc_loss = 0
        total_vtx_loss = 0
        for i in range(BATCHSIZE):
            # get the batch index of the current batch
            pfc_indices = (data.x_pfc_batch == i)
            vtx_indices = (data.x_vtx_batch == i)
            # get the embedding of the pfc, vtx, and truth in the current batch
            pfc_enc_batch = pfc_enc[pfc_indices, :]
            vtx_enc_batch = vtx_enc[vtx_indices, :]
            truth_batch = data.truth[pfc_indices].to(dtype=torch.int64, device=device)
            # get length of vtx_enc_batch
            vtx_enc_batch_len = vtx_enc_batch.shape[0]
            # pop out truth values which are greater than the length of vtx_enc_batch or less than 0
            valid_vertices = (truth_batch < vtx_enc_batch_len) & (truth_batch >= 0)
            truth_batch = truth_batch[valid_vertices]
            pfc_enc_batch = pfc_enc_batch[valid_vertices, :]
            # the true encoding is the embedding of the true vertex
            true_pfc_encoding = vtx_enc_batch[truth_batch, :]
            # print(true_pfc_encoding.shape)
            # print(true_pfc_encoding.device)
            # print(true_pfc_encoding)
            # raise(Exception("stop"))
            # check if the tensors are on the same device
            # print(pfc_enc_batch.device, true_pfc_encoding.device)
            # print(pfc_enc_batch.shape, true_pfc_encoding.shape)
            # print(pfc_enc_batch, true_pfc_encoding)
            # raise(Exception("stop"))
            # pfc_loss = euclidean_loss(pfc_enc_batch[:], true_pfc_encoding[:])
            # raise(Exception("pfc_loss: ", pfc_loss))
            # the loss is the MSE distance between the embedding of the pfc and the true vertex
            pfc_loss = euclidean_loss(pfc_enc_batch, true_pfc_encoding)
            # add the loss to the total loss
            # print("Particle loss: ", pfc_loss)
            total_pfc_loss += pfc_loss
            # calculate the loss due to the vtx embedding
            # the vertex embedding should be as far from other vertices as possible
            # the loss is the MSE distance between the embedding of the vtx and the other vertices
            # randomly choose 25 vertices to calculate the loss
            random_indices = torch.randperm(vtx_enc_batch_len)[:25]
            random_vtx_encoding = vtx_enc_batch[random_indices, :]
            for j in range(len(random_vtx_encoding)):
                for k in range(j+1, len(random_vtx_encoding)):
                    vtx_loss = -0.001*euclidean_loss(random_vtx_encoding[j, :], random_vtx_encoding[k, :])
                    total_vtx_loss += vtx_loss
        # regularization loss
        reg_loss = 10*((torch.norm(vtx_enc, p=2, dim=1) - 10)**2).mean()
        print(torch.norm(vtx_enc, p=2, dim=1).mean())
        print("Particle loss: ", total_pfc_loss)
        print("Vertex loss: ", total_vtx_loss)
        print("Reg loss: ", reg_loss)
        loss = total_pfc_loss + total_vtx_loss + reg_loss
        print("loss after regularization: ", loss)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if counter % 100 == 0:
            print("Iteration: ", counter, " Loss: ", total_loss)
    total_loss = total_loss / counter        
    return total_loss

# test function
@torch.no_grad()
def test():
    upuppi.eval()
    euclidean_loss = nn.MSELoss()
    counter = 0
    loss = 0
    for data in test_loader:
        counter += 1
        data = data.to(device)
        pfc_enc, vtx_enc = upuppi(data.x_pfc, data.x_vtx, data.x_pfc_batch, data.x_vtx_batch)
        for i in range(len(BATCHSIZE)):
            pfc_indices = (data.x_pfc_batch == i)
            pfc_enc_batch = pfc_enc[pfc_indices, :]
            truth_batch = data.truth[pfc_indices]
            vtx_indices = (data.x_vtx_batch == i)
            vtx_enc_batch = vtx_enc[vtx_indices, :]
            true_pfc_encoding = vtx_enc_batch[truth_batch, :]
            pfc_loss = euclidean_loss(pfc_enc_batch, true_pfc_encoding)
            loss += pfc_loss.item()
            for j in range(len(vtx_enc_batch)):
                for k in range(j+1, len(vtx_enc_batch)):
                    vtx_loss = euclidean_loss(vtx_enc_batch[j, :], vtx_enc_batch[k, :])
                    loss += vtx_loss.item()
    loss = loss / counter
    return loss

# train the model

for epoch in range(10):
    loss = 0
    test_loss = 0
    print("Epoch: ", epoch, " Loss: ", loss, " Test Loss: ", test_loss)
    state_dicts = {'model':upuppi.state_dict(),
                   'opt':optimizer.state_dict()} 

    torch.save(state_dicts, os.path.join(model_dir, 'epoch-{}.pt'.format(epoch)))
    print("Model saved")
    print("Time elapsed: ", time.time() - start_time)
    print("-----------------------------------------------------")
    loss = train()
    test_loss = test()


import time, os, torch, numpy as np
from upuppi_v0_dataset import UPuppiV0
from torch_geometric.data import DataLoader
from torch import nn
from torch.nn import functional as F
from models.embedding_model import Net
from tqdm import tqdm


BATCHSIZE = 32
start_time = time.time()
print("Training...")
data_train = UPuppiV0("/work/submit/cfalor/upuppi/deepjet-geometric/train/")
data_test = UPuppiV0("/work/submit/cfalor/upuppi/deepjet-geometric/test/")


train_loader = DataLoader(data_train, batch_size=BATCHSIZE, shuffle=True,
                          follow_batch=['x_pfc', 'x_vtx'])
test_loader = DataLoader(data_test, batch_size=BATCHSIZE, shuffle=True,
                         follow_batch=['x_pfc', 'x_vtx'])

model = "contrastive_loss"
model_dir = '/work/submit/cfalor/upuppi/deepjet-geometric/models/{}/'.format(model)
#model_dir = '/home/yfeng/UltimatePuppi/deepjet-geometric/models/v0/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
# print the device used
print("Using device: ", device, torch.cuda.get_device_name(0))

# create the model
net = Net().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)


def contrastive_loss(pfc_enc, vtx_id, num_pfc=128, c=1.0):
    '''
    Calculate the contrastive loss
    input:
    pfc_enc: the encodding of the inputs
    vtx_id: the true vertex which the particle is connected to
    num_pfc: number of particles to randomly sample
    c: the ratio of positive factor for the particles of same vertex divided by the negative factor
    '''
    # loss which encourages the embedding of same particles to be close and different particles to be far
    # randomly select a set of particles to be used for contrastive loss
    random_perm = torch.randperm(len(pfc_enc))
    random_indices1 = random_perm[:num_pfc]
    random_indices2 = random_perm[num_pfc:2*num_pfc]
    pfc_enc_1 = pfc_enc[random_indices1, :]
    pfc_enc_2 = pfc_enc[random_indices2, :]
    vtx_id_1 = vtx_id[random_indices1]
    vtx_id_2 = vtx_id[random_indices2]
    # get a mask which is 1 if the particles are the same and -1 if they are different
    mask = -1+(c+1)*(vtx_id_1 == vtx_id_2).float()
    euclidean_dist = F.pairwise_distance(pfc_enc_1, pfc_enc_2)
    loss = torch.mean(mask*torch.pow(euclidean_dist, 2))
    return loss





def train(reg_ratio = 0.01):
    net.train()
    train_loss = 0
    for counter, data in enumerate(tqdm(train_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        pfc_enc = net(data.x_pfc)
        vtx_id = (data.truth != 0).int()
        loss = contrastive_loss(pfc_enc, vtx_id)
        loss += reg_ratio*((torch.norm(pfc_enc, p=2, dim=1)/10)**4).mean()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if counter % 100 == 1:
            print("Counter: {}, Average Loss: {}".format(counter, train_loss/counter))
            print("Regression loss: {}".format(((torch.norm(pfc_enc, p=2, dim=1)/10)**4).mean()))
    train_loss = train_loss/counter
    return train_loss

# test function
@torch.no_grad()
def test():
    net.eval()
    test_loss = 0
    for counter, data in enumerate(train_loader):
        data = data.to(device)
        pfc_enc = net(data.x_pfc)
        vtx_id = data.truth
        loss = contrastive_loss(pfc_enc, vtx_id)
        test_loss += loss.item()
    test_loss = test_loss / counter        
    return test_loss

# train the model

for epoch in range(20):
    loss = 0
    test_loss = 0
    loss = train()
    state_dicts = {'model':net.state_dict(),
                   'opt':optimizer.state_dict()} 
    torch.save(state_dicts, os.path.join(model_dir, 'epoch-{}.pt'.format(epoch)))
    print("Model saved")
    print("Time elapsed: ", time.time() - start_time)
    print("-----------------------------------------------------")
    # test_loss = test()
    print("Epoch: ", epoch, " Loss: ", loss, " Test Loss: ", test_loss)

    


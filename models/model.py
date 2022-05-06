import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.pool import avg_pool_x
from torch.nn import Sequential, Linear


class Net(nn.Module):
    def __init__(self, isFCN=False):
        self.isFCN = isFCN

        super(Net, self).__init__()
        hidden_dim = 32
        self.vtx_encode = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )

        #self.glob_encode = nn.Sequential(
        #    nn.Linear(1, hidden_dim),
        #    nn.ELU(),
        #    nn.Linear(hidden_dim, hidden_dim),
        #    nn.ELU()
        #)

        if not self.isFCN:
            self.pfc_encode = nn.Sequential(
                nn.Linear(7, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU()
            )

            self.conv = DynamicEdgeConv(
                nn=nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.LeakyReLU()),
                k=16
            )

            self.output = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 4),
                nn.LeakyReLU(),
                nn.Linear(4, 1), nn.LeakyReLU()
            )
        else:
            self.pfc_encode = nn.Sequential(
                nn.Linear(7, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 1)
            )


    def forward(self,
                x_pfc, x_vtx,
                batch_pfc, batch_vtx):
        batch = batch_pfc

        if not self.isFCN:
            x_pfc_enc = self.pfc_encode(x_pfc)
            x_vtx_enc = self.vtx_encode(x_vtx)
            #x_glob_enc = self.glob_encode(x_glob)

            # create a representation of PFs to clusters
            feats1 = self.conv(x=(x_pfc_enc, x_pfc_enc), batch=(batch_pfc, batch_pfc))

            # similarly a representation of PFs-clusters amalgam to PFs
            feats2 = self.conv(x=(x_vtx_enc, feats1), batch=(batch_vtx, batch_pfc))

            # now to global variables
            #feats3 = self.conv(x=(x_glob_enc, feats2), batch=(batch_pfc, batch_pfc))

            #out, batch = avg_pool_x(batch, feats2, batch)        
            #out = self.output(out)
            out = self.output(feats2)
            #out = self.output(x_pfc_enc)
        else:
            out = self.pfc_encode(x_pfc)

        return out, batch


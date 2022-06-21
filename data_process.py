# opens a file with 1000 events and processes data to a neural network friendly format

import h5py
import sys
import torch
import os
import numpy as np
from tqdm import tqdm
# check 48 for error
# if __name__ == '__main__':
#     file = h5py.File("/work/submit/cfalor/upuppi/z_reg/train/notr/samples_v0_dijet_48.h5", "r")
#     file_out = h5py.File("/work/submit/cfalor/upuppi/z_reg/train/raw/samples_v0_dijet_48.h5", "w")
for fileid in range(1, 50):
#     if fileid == 40:
#         file = h5py.File("/work/submit/cfalor/upuppi/z_reg/train/notr/samples_v0_dijet_4.h5", "r")
#         # make a new file to store the processed data
#         file_out = h5py.File("/work/submit/cfalor/upuppi/z_reg/train/raw/samples_v0_dijet_4.h5", "w")
    # if fileid == 56:
    #     file = h5py.File("/work/submit/cfalor/upuppi/z_reg/test/notr/samples_v0_dijet_5.h5", "r")
    #     # make a new file to store the processed data
    #     file_out = h5py.File("/work/submit/cfalor/upuppi/deepjet-geometric/test/raw/samples_v0_dijet_5.h5", "w")
#     else:
    try:
        # file = h5py.File("/work/submit/cfalor/upuppi/z_reg/train/notr/samples_v0_dijet_"+str(fileid)+".h5", "r")
        file = h5py.File("/work/submit/bmaier/upuppi/data/v0_z_regression_pu30/train/raw/samples_v0_dijet_"+str(fileid)+".h5", "r")
        file_out = h5py.File("/work/submit/cfalor/upuppi/deepjet-geometric/train2/raw/samples_v0_dijet_"+str(fileid)+".h5", "w")
    except:
        print("fileid:", fileid)
        continue
    # copy the header from the original file
    for key in file.keys():
        if key == "vtx" or key == "truth":
            file_out.create_dataset(key, data=file[key][:])
        elif key=="n":
            # it is the number of events, scalar
            file_out.create_dataset(key, data=file[key])

    # process the data
    # pfs: coordinates of particles in the event
    # pfs_shape: (1000, 7000, 7)
    # pt, eta, phi, E, pid, charge, z-position for pfs

    with file as f:
        pfs = f["pfs"][:]

        # one hot encode the pid
        # pid takes values 0, 1, 2, 3, 4, -13, 13
        # pid_onehot takes values 0, 1, 2, 3, 4, 5, 6 respectively (one hot encoded)
        # initialize the pid_onehot array
        pid = np.zeros((pfs.shape[0], pfs.shape[1], 7))
        for i in tqdm(range(pfs.shape[0])):
            for j in range(pfs.shape[1]):
                particle_id = int(pfs[i,j,4])
                if particle_id <= -13:
                    pid[i,j,5] = 1
                elif particle_id == 13:
                    pid[i,j,6] = 1
                else:
                    try:
                        pid[i,j,particle_id] = 1
                    except:
                        print(i, j, particle_id)
                        raise(Exception("error"))

        # convert pt, phi to cartesian coordinates
        # pt: (1000, 7000)
        # phi: (1000, 7000)
        # px: (1000, 7000)
        # py: (1000, 7000)
        # convert torch tensor to numpy array
        pt = pfs[:,:,0]
        phi = pfs[:,:,2]
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        x = np.cos(phi)
        y = np.sin(phi)

        z = pfs[:,:,6]
        # normalize the z-position
        z = z / 200

        
        # pt, eta, phi, E, pid, charge, z-position for pfs
        # save the processed data in pfs
        new_pfs = np.concatenate((x[:,:,np.newaxis], y[:,:,np.newaxis], pfs[:,:,1:2], z[:,:,np.newaxis], pid, pfs[:,:,5:6]), axis=2)
        # cos(phi), sin(phi), eta, z, pid, charge for pfs
        # print out the shape of the data
        print("new_pfs shape:", new_pfs.shape)
        ztrue = f["truth"][:]
        ztrue = ztrue / 200

    # save the processed data in the file
    file_out.create_dataset("pfs", data=new_pfs)
    file_out.create_dataset("z", data=ztrue)


    # save the file
    file_out.close()

    # close the original file
    file.close()
    # clean up

  
    


import numpy as np 
import h5py
import urllib
import os
import matplotlib.pyplot as plt
import urllib.request
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import tqdm

from test_python.kaczmarzReg import *
from test_python.pseudoinverse import *

from MPI_utils.weight_matrix import *
from MPI_utils.initailize_lambda import *
from MPI_utils.time import *
from MPI_utils.computation_with_timeout import *

SM_path = '.\\OpenMPI\\calibrations\\6.mdf'
Measurment_path = '.\\OpenMPI\\measurements\\concentrationPhantom\\3.mdf'
fSM = h5py.File(SM_path, 'r')
fMeas = h5py.File(Measurment_path, 'r')

SM = fSM['/measurement/data']
print(SM.shape) # (1*3*26929*52023) 
U = fMeas['/measurement/data']
print(U.shape) # (2000*1*3*53856)
U = U[:,:,:,:].squeeze()
U = np.fft.rfft(U,axis=2)
print(U.shape) # (2000*3*26969)

isBG = fSM['/measurement/isBackgroundFrame'][:].view(bool)
BG = np.sum(isBG) 
print(BG) # 1370

# select frequency range according to DFT
sampling_num = fMeas['/acquisition/receiver/numSamplingPoints'][()] #53856
frequency_num = round(sampling_num / 2) + 1 #26929
bandwidth = fMeas['/acquisition/receiver/bandwidth'][()] #1250000
freq_range = np.arange(0, frequency_num) / (frequency_num-1) * bandwidth
min_freq = np.where(freq_range > 80e3)[0][0] #1724

U = U[:, 0:2, min_freq:] # (2000*2*25205)
U = np.reshape(U,(U.shape[0],U.shape[1]*U.shape[2]))
U = np.mean(U,axis=0) # (50410,)

energy_per_row,norm_matrix,identity_matrix = np.zeros(U.shape),np.zeros(U.shape),np.ones(U.shape)
for i in tqdm(range(int(len(U)/2)),desc="calculating energy per rows"): # about 4 min
    energy_per_row[i] = np.linalg.norm(SM[0,0,i+min_freq,0:SM.shape[3]-BG])
    energy_per_row[i+int(len(U)/2)] = np.linalg.norm(SM[0,1,i+min_freq,0:SM.shape[3]-BG])
for i in tqdm(range(len(U)),desc="calculating normalization matrix"):
    norm_matrix[i] = np.ones(()) / energy_per_row[i] ** 2

def Kaczmarz_weight(num_epoch,reg_lambda,weight_matrix,is_shuffle):
    M = len(U)
    N = SM.shape[3] - BG
    print(M,N)

    X = np.zeros(N,dtype = np.complex128)
    V = np.zeros(M,dtype = np.complex128)
    RowIndex = np.arange(M)

    for i in tqdm(range(num_epoch),desc = "reconstruction cycle"):
        if is_shuffle:
            np.random.shuffle(RowIndex)
        for j in tqdm(range(M),desc=f"cycle per epoch{i+1}"):
            k = RowIndex[j]
            channel_choice = 1 if k >= M/2 else 0
            if k < M/2:
                alpha = (U[k] - np.dot(SM[0,channel_choice,min_freq+k,0:N],X) - np.sqrt(reg_lambda / weight_matrix[k]) * V[k]) / (energy_per_row[k] ** 2 + reg_lambda / weight_matrix[k])
                X += alpha * SM[0,channel_choice,min_freq+k,0:N].conjugate()
            else:
                alpha = (U[k] - np.dot(SM[0,channel_choice,min_freq+k-M//2,0:N],X) - np.sqrt(reg_lambda / weight_matrix[k]) * V[k]) / (energy_per_row[k] ** 2 + reg_lambda / weight_matrix[k])
                X += alpha * SM[0,channel_choice,min_freq+k-M//2,0:N].conjugate()
            V[k] += alpha * np.sqrt(reg_lambda / weight_matrix[k])

    X.imag = 0
    X = X * (X.real > 0)
    return X

origin_lambda = 0.01
weight_mean = np.sqrt(np.mean(norm_matrix))
matrix_Frob = np.linalg.norm(SM[0,0,min_freq:min_freq+100,:])
reg_lambda = origin_lambda * weight_mean * matrix_Frob
print(origin_lambda,weight_mean,matrix_Frob,reg_lambda)

C0 = Kaczmarz_weight(num_epoch=2,reg_lambda=0,weight_matrix=norm_matrix,is_shuffle=False)

def ImageReshape(c,size,type):
    #reshape之后的顺序和实际采集得到的x-y-z顺序不一致
    x = size[0]
    y = size[1]
    z = size[2]
    c_origin = np.real(np.reshape(c,(z,y,x)))

    reshape_xy = np.zeros((z,y,x))# x*y*z
    reshape_yz = np.zeros((x,z,y)) # y*z*x
    reshape_xz = np.zeros((y,z,x)) # x*z*y
    for i in range(z):
        for j in range(y):
            for k in range(x):
                reshape_xy[i,j,k] = c_origin[i,j,k] # x=k,y=j,z=i
                reshape_yz[k,i,j] = c_origin[i,j,k]
                reshape_xz[j,i,k] = c_origin[i,j,k]
    if type:
        Ixy = np.max(reshape_xy,axis=0)
        Ixz = np.max(reshape_xz,axis=0)
        Iyz = np.max(reshape_yz,axis=0)

    return reshape_xy,reshape_yz,reshape_xz,np.transpose(Ixy/np.max(Ixy)),Iyz/np.max(Iyz),Ixz/np.max(Ixz)

reshape_xy,reshape_yz,reshape_xz,Ixy,Iyz,Ixz = ImageReshape(C0,(37,37,37),True)

fig, axes = plt.subplots(1,3,figsize = (8,3))
axes = axes.flatten()

print(np.max(reshape_xy[6,:,:]))
print(np.max(reshape_xz[6,:,:]))
print(np.max(reshape_yz[6,:,:]))


# im1 = axes[0].imshow(Ixy, cmap="plasma", vmin=0,vmax=1)
# axes[0].set_title("Mip xy")
# im2 = axes[1].imshow(Ixz, cmap="plasma", vmin=0,vmax=1)
# axes[1].set_title("Mip xz")
# im3 = axes[2].imshow(Iyz, cmap="plasma", vmin=0,vmax=1)
# axes[2].set_title("Mip yz")

im1 = axes[0].imshow(reshape_xy[6,:,:].transpose(), cmap="plasma", vmin=0,vmax=0.0012)
axes[0].set_title("Slice at z=7 xy")
im2 = axes[1].imshow(reshape_xz[6,:,:], cmap="plasma", vmin=0,vmax=0.0012)
axes[1].set_title("Slice at y=7 xz")
im3 = axes[2].imshow(reshape_yz[6,:,:], cmap="plasma", vmin=0,vmax=0.0012)
axes[2].set_title("Slice at x=7 yz")

cbar = fig.colorbar(im3,ax=axes,orientation = 'vertical',fraction = 0.04,pad=0.04)

os.makedirs(".\\result_path\\OpenMPI\\Concentration",exist_ok=True)
plt.savefig(os.path.join(".\\result_path\\OpenMPI\\Concentration","3D_6_norm.png"))
plt.show()


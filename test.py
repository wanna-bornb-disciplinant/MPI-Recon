import os 
import numpy
import sys
import h5py
import tqdm

from matplotlib import pyplot as plt
from BaseClass.Constant_Base import *
from Phantom.Shape_1_Phantom import * 
from Phantom.P_Phantom import *

# SM_path1 = "OpenMPI/measurements/shapePhantom/1.mdf"
# SM_path2 = "OpenMPI/measurements/shapePhantom/2.mdf"
# SM_path3 = "OpenMPI/measurements/shapePhantom/3.mdf"

# SM_path_con_path1 = "OpenMPI/measurements/concentrationPhantom/1.mdf"
# SM_path_con_path2 = "OpenMPI/measurements/concentrationPhantom/2.mdf"
# SM_path_con_path3 = "OpenMPI/measurements/concentrationPhantom/3.mdf"

# SM_path_reso1 = "OpenMPI/measurements/resolutionPhantom/1.mdf"
# SM_path_reso2 = "OpenMPI/measurements/resolutionPhantom/2.mdf"
# SM_path_reso3 = "OpenMPI/measurements/resolutionPhantom/3.mdf"

# SM_path_rota1 = "OpenMPI/measurements/rotationPhantom/1.mdf"
# SM_path_rota2 = "OpenMPI/measurements/rotationPhantom/2.mdf"
# SM_path_rota3 = "OpenMPI/measurements/rotationPhantom/3.mdf"
# SM_path_rota4 = "OpenMPI/measurements/rotationPhantom/4.mdf"

# Cali_path0 = "OpenMPI/calibrations/1.mdf"
# Cali_path1 = "OpenMPI/calibrations/2.mdf"
# Cali_path2 = "OpenMPI/calibrations/3.mdf"
# Cali_path3 = "OpenMPI/calibrations/4.mdf"
# Cali_path4 = "OpenMPI/calibrations/5.mdf"
# Cali_path5 = "OpenMPI/calibrations/6.mdf"
# Cali_path6 = "OpenMPI/calibrations/7.mdf"
# Cali_path7 = "OpenMPI/calibrations/8.mdf"
# Cali_path8 = "OpenMPI/calibrations/9.mdf"
# Cali_path9 = "OpenMPI/calibrations/10.mdf"
# Cali_path10 = "OpenMPI/calibrations/11.mdf"
# Cali_path11 = "OpenMPI/calibrations/12.mdf"
# Cali_path12 = "OpenMPI/calibrations/13.mdf"
# Cali_path13 = "OpenMPI/calibrations/14.mdf"
# Cali_path14 = "OpenMPI/calibrations/15.mdf"
# Cali_path15 = "OpenMPI/calibrations/16.mdf"
# Cali_path16 = "OpenMPI/calibrations/17.mdf"


# # with h5py.File(SM_path, 'r') as f:
# #     def print_datasets(name, obj):
# #         if isinstance(obj, h5py.Dataset):
# #             pass
# #             #print(f"Dataset: {name}")
# #         elif isinstance(obj, h5py.Group):
# #             print(f"Group: {name}")

# #     f.visititems(print_datasets)

# # with h5py.File(Cali_path,"r") as f:
# #     def print_datasets(name,obj):
# #         if isinstance(obj, h5py.Dataset):pass
# #             #print(f"Dataset: {name}")
# #         elif isinstance(obj, h5py.Group):
# #             print(f"Group: {name}")

# #     f.visititems(print_datasets)

# for i in range(17):
#     file = eval(f"Cali_path{i}")
#     print(h5py.File(file,"r")[MDF_data].shape,end = " ")
#     print(h5py.File(file,"r")[MDF_sample_num][()],end = " ")
#     print(h5py.File(file,"r")[MDF_voxel_number][()],end = " ")
#     print(h5py.File(file,"r")[MDF_CALI_FOV][()],end = " ")
#     print(h5py.File(file,"r")['/calibration/order'][()],end = " ")
#     sum = 0
#     is_background = h5py.File(file,"r")[MDF_isBackground][:].view(bool)
#     for j in range(len(is_background)):
#         if is_background[j] == True:
#             sum += 1
#     print(sum)

#     print(np.where(is_background != 0)[0][0])

# print("-"*10) 

# for i in range(3):
#     file = eval(f"SM_path{i+1}")
#     print(h5py.File(file,"r")[MDF_data].shape,end = " ")
#     print(h5py.File(file,"r")[MDF_sample_num][()],end = " ")
#     sum = 0
#     is_background = h5py.File(file,"r")[MDF_isBackground][:].view(bool)
#     print(len(is_background),end=" ")
#     for j in range(len(is_background)):
#         if is_background[j] == True:
#             sum += 1
#     print(sum) #shape

# print("-"*10)

# for i in range(3):
#     file = eval(f"SM_path_con_path{i+1}")
#     print(h5py.File(file,"r")[MDF_data].shape,end = " ")
#     print(h5py.File(file,"r")[MDF_sample_num][()],end = " ")
#     sum = 0
#     is_background = h5py.File(file,"r")[MDF_isBackground][:].view(bool)
#     print(len(is_background),end=" ")
#     for j in range(len(is_background)):
#         if is_background[j] == True:
#             sum += 1 
#     print(sum) #concentration

# print("-"*10)

# for i in range(3):
#     file = eval(f"SM_path_reso{i+1}")
#     print(h5py.File(file,"r")[MDF_data].shape,end = " ")
#     print(h5py.File(file,"r")[MDF_sample_num][()],end = " ")
#     sum = 0
#     is_background = h5py.File(file,"r")[MDF_isBackground][:].view(bool)
#     print(len(is_background),end=" ")
#     for j in range(len(is_background)):
#         if is_background[j] == True:
#             sum += 1 
#     print(sum) #resolution

# print("-"*10)

# for i in range(4):
#     file = eval(f"SM_path_rota{i+1}")
#     print(h5py.File(file,"r")[MDF_data].shape,end = " ")
#     print(h5py.File(file,"r")[MDF_sample_num][()],end = " ")
#     sum = 0
#     is_background = h5py.File(file,"r")[MDF_isBackground][:].view(bool)
#     print(len(is_background),end=" ")
#     for j in range(len(is_background)):
#         if is_background[j] == True:
#             sum += 1 
#     print(sum) #rotatio

import numpy as np
from matplotlib import pyplot as plt

class Weight_Matrix:

    '''
    calculate the weight matrix for system matrix
    
    --Paras:
    '''
    def __init__(self,A):
        self.energy = self.row_energy(A)
        self.Matrix_size = A.shape[0]

    def row_energy(self,A):
        M = A.shape[0]
        energy = np.zeros(M, dtype = np.double)
        for m in range(M):
            energy[m] = np.linalg.norm(A[m,:]) 
        return energy

    def identity_matrix(self): 
        return np.ones(self.Matrix_size)

    def energy_matrix(self):
        return np.ones_like(self.energy) * self.energy

    def normalization_matrix(self):
        return np.ones_like(self.energy) / self.energy ** 2

def Kaczmarz_origin(A,b,weight_matrix,Lambda,is_Shuffle):
        energy = weight_matrix.energy

        M = A.shape[0]
        N = A.shape[1]

        x = np.zeros(N, dtype = np.complex128)
        v = np.zeros(M, dtype = np.complex128)

        rowIndexCycle = np.arange(0,M)
        if is_Shuffle:
            np.random.shuffle(rowIndexCycle)

        for i in range(10):
            for j in range(M):
                k = rowIndexCycle[j]
                alpha = (b[0][k] - np.dot(A[k,:],x) - np.sqrt(Lambda) * v[k]) / (energy[k] ** 2 + Lambda)

                x += alpha * A[k,:].conjugate()
                v[k] += alpha * np.sqrt(Lambda)
        
        x.imag = 0
        x = x * (x.real > 0)

        return x

def Kaczmarz_weight(A,b,weight_matrix,Lambda,is_Shuffle):
        energy = weight_matrix.energy
        matrix = weight_matrix.energy_matrix()

        M = A.shape[0]
        N = A.shape[1]

        x = np.zeros(N, dtype = np.complex128)
        v = np.zeros(M, dtype = np.complex128)

        rowIndexCycle = np.arange(0,M)
        if is_Shuffle:
            np.random.shuffle(rowIndexCycle)

        for i in range(10):
            for j in range(M):
                k = rowIndexCycle[j]
                alpha = (b[0][k] - np.dot(A[k,:],x) - np.sqrt(Lambda / matrix[k]) * v[k]) / (energy[k] ** 2 + Lambda / matrix[k])

                x += alpha * A[k,:].conjugate()
                v[k] += alpha * np.sqrt(Lambda / matrix[k])
        
        x.imag = 0
        x = x * (x.real > 0)

        return x

label = np.array(np.arange(100)) * 0.1
SM_1 = np.random.randn(200,100) * 1e-23
SM_2 = np.random.randn(200,100) * 1e-23
SM = SM_1 + 1j * SM_2

data = SM@label.reshape(-1,1)
data = data.reshape(1,-1)

norm_factor = np.linalg.norm(SM,axis=0)
max_norm = np.max(norm_factor)
if max_norm < 1e-12:
    print("⚠️ 警告：系统矩阵范数太小，可能存在数值不稳定风险。")
scale_factor = 1 / max_norm
SM *= scale_factor
data *= scale_factor

weight_matrix = Weight_Matrix(SM)
matrix_pre = weight_matrix.energy_matrix()
lambda_init = 0.01
# print(lambda_init * np.linalg.norm(SM))
SM_condition = np.zeros_like(SM,dtype=SM.dtype)
for i in range(len(matrix_pre)):
     SM_condition[i,:] = np.sqrt(matrix_pre[i]) * SM[i,:]
lambda_matrix = lambda_init * np.linalg.norm(SM_condition)
# print(lambda_matrix)

recon_label = Kaczmarz_weight(SM,data,weight_matrix,lambda_matrix,False)

print(recon_label)
# print(SM_norm)
# print(data_norm)
# recon_label = recon_label*SM_norm/data_norm
# print(recon_label)

# recon_label *= SM_norm / data_norm


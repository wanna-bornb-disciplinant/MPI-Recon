from email import message
import numpy as np
from tqdm import tqdm
from zmq import Message

from BaseClass.Constant_Base import *

from MPI_utils.weight_matrix import *
from MPI_utils.time import *
from MPI_utils.computation_with_timeout import *


class SM_recon_Kaczmarz(object):
    def __init__(self,Message,Weight_type,is_Shuffle,Recon_type=2,Iterations=10,Lambda=1e-2): 
        '''
        Reconstruction method based on Kacmarz 
        reconstruction problem is easily considered as the inconsistant linear problem, so when we use kacmarz method we need to apply the Kacmarz Iteration to the extend system,
        introduced 'v'

        Attributes:
            -- Recon_type: 1--no weight matrix, 2--with weight matrix
            -- Iterations: the number of iterations
            -- Enforce_Rule: whether enforce the concentration to be positive and no imaginary part
            -- Weight_type: 1--identity matrix, 2--energy matrix, 3--normalization matrix
            -- Lambda: the regularization parameter
            -- is_Shuffle: random Kaczmarz or sequential Kaczmarz
        '''

        self._Image_data = []
        self._Iterations = Iterations
        self._is_Shuffle = is_Shuffle
        self._num_epoch = Message[measurement][auxiliary_information].shape[0]

        self.selection_matrix = Weight_Matrix(Message[measurement][auxiliary_information])

        timer = Time() 

        if Weight_type == 1:
            self._Matrix = self.selection_matrix.identity_matrix()
        elif Weight_type == 2:
            self._Matrix = self.selection_matrix.energy_matrix()
        elif Weight_type == 3:
            self._Matrix = self.selection_matrix.normalization_matrix()
        else:
            raise Exception("Weight_type is wrong !!!")        
             
        self._is_adjust,self._scale_factor = self._examine_scale(Message[measurement][auxiliary_information],Message[measurement][measure_signal])

        if Recon_type == 1:
            self._ImageRecon1(Message[measurement][auxiliary_information],Message[measurement][measure_signal],Message[measurement][voxel_number],Message[measurement][voxel_size],Lambda)
        else:
            self._ImageRecon2(Message[measurement][auxiliary_information],Message[measurement][measure_signal],Message[measurement][voxel_number],Message[measurement][voxel_size],Lambda)
        
        timer.cal_time()
        print(f"reconstrution time: ", timer.time[-1])
        print(self._Lambda)
        timer.reset()  

    def _examine_scale(self,A,b):
        scale_factor = np.zeros([])
        norm_factor = np.linalg.norm(A,axis=0)
        norm_max_factor = np.max(norm_factor)
        if norm_max_factor < 1e-12:
            print("⚠️ 警告：系统矩阵范数太小，可能存在数值不稳定风险。") 
            is_adjust = True
        else:
            is_adjust = False 
        scale_factor = 1 / norm_max_factor

        return is_adjust,scale_factor


    def _Kaczmarz_origin(self,A,b,voxel_size,Lambda):
        '''
        random choose the row of system matrix 
        not enforce the concentration to be positive and no imaginary part
        no weight matrix and some tricks about denoising
        
        Parameters:
            --A: m*n
            --b: 1*m
        '''
        if self._is_adjust:
            A *= self._scale_factor
            b *= self._scale_factor

        self._Lambda = Lambda * np.linalg.norm(A,ord = "fro")

        energy = self.selection_matrix.energy
        M = A.shape[0]
        N = A.shape[1]

        x = np.zeros(N, dtype = np.complex128)
        v = np.zeros(M, dtype = np.complex128)

        rowIndexCycle = np.arange(0,M)
        if self._is_Shuffle:
            np.random.shuffle(rowIndexCycle)

        for i in tqdm(range(self._Iterations),desc="Reconstruction Calculation"):
            for j in range(M):
                k = rowIndexCycle[j]
                alpha = (b[0][k] - np.dot(A[k,:],x) - np.sqrt(self._Lambda / self._Matrix[k]) * v[k]) / (energy[k] ** 2 + self._Lambda / self._Matrix[k])

                x += alpha * A[k,:].conjugate()
                v[k] += alpha * np.sqrt(self._Lambda / self._Matrix[k])
        
        x.imag = 0
        x = x * (x.real > 0)

        return x / voxel_size

    def _Kaczmarz_withweight(self,A,b,voxel_size,Lambda):
        '''
        random choose the row of system matrix 
        can choose whether enforce the concentration to be positive and no imaginary part
        has weight matrix and some tricks about denoising'''

        if self._is_adjust:
            A *= self._scale_factor
            b *= self._scale_factor

        energy = self.selection_matrix.energy
        temp_aux = np.zeros_like(A,dtype=A.dtype)
        for i in range(self._num_epoch):
            temp_aux[i,:] = A[i,:] * np.sqrt(self._Matrix[i])
        self._Lambda = Lambda * np.linalg.norm(temp_aux,ord = "fro")

        M = A.shape[0]
        N = A.shape[1]

        x = np.zeros(N, dtype = b.dtype)
        v = np.zeros(M, dtype = b.dtype)

        rowIndexCycle = np.arange(0,M)
        if self._is_Shuffle:
            np.random.shuffle(rowIndexCycle)

        for i in tqdm(range(self._Iterations),desc="Reconstruction Calculation"):
            for j in range(M):
                k = rowIndexCycle[j]
                alpha = (np.sqrt(self._Matrix[k]) * b[0][k] - np.sqrt(self._Matrix[k]) * np.dot(A[k,:],x) - np.sqrt(self._Lambda) * v[k]) / (self._Matrix[k] * energy[k]**2 + self._Lambda)

                x += alpha * np.sqrt(self._Matrix[k]) * A[k,:].conjugate()
                v[k] += alpha * np.sqrt(self._Lambda)
                # if self._Enforce_Rule:
                #     if np.iscomplexobj(x):
                #         x.imag = 0
                #     x = x * (x.real > 0)
        
        x.imag = 0
        x = x * (x.real > 0)
        return x / voxel_size


    def _ImageReshape(self,c,size):
        y = size[0]
        x = size[1]
        # c = np.flip(c)
        c = np.real(np.reshape(c,(y,x)))
        c /= np.max(c) 
        return c
    
    def _ImageRecon1(self,A,b,size,voxel_size,Lambda):
        origin_x = self._Kaczmarz_origin(A,b,voxel_size,Lambda)
        self._Image_data.append(self._ImageReshape(origin_x,size))
    
    def _ImageRecon2(self,A,b,size,voxel_size,Lambda):
        origin_x = self._Kaczmarz_withweight(A,b,voxel_size,Lambda)
        self._Image_data.append(self._ImageReshape(origin_x,size))

    def get_Image(self):
        return self._Image_data[-1]    

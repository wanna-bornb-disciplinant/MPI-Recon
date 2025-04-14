import numpy as np
from tqdm import tqdm

from BaseClass.Constant_Base import *

from MPI_utils.weight_matrix import *
from MPI_utils.time import *
from MPI_utils.computation_with_timeout import *


class SM_recon_CGNR(object):
    def __init__(self,Message,Weight_type,Iterations=10,Lambda=1e-3): 
        '''
        Reconstruction method based on Conjugate Gradient Normal Residual

        Attributes:
            -- Message: data needed in reconstruction 
            -- Weight_type: 1--energy matrix, 2--identity matrix, 3--normalization matrix
            -- Lambda: the regularization parameter
        '''

        self._Image_data = []
        self._Iterations = Iterations
        self._Lambda = Lambda * np.linalg.norm(Message[measurement][auxiliary_information],ord="fro")

        timer = Time() 

        if Weight_type == 1:
            self._Matrix = energy_matrix(Message[measurement][auxiliary_information])
        elif Weight_type == 2:
            self._Matrix = identity_matrix(Message[measurement][auxiliary_information])
        elif Weight_type == 3:
            self._Matrix = normalization_matrix(Message[measurement][auxiliary_information])
        else:
            raise Exception("Weight_type is wrong !!!")
        
        timer.cal_time()
        print(f"matrix_time", timer.time[-1])
        timer.begin_time()

        self._ImageRecon(Message[measurement][auxiliary_information],Message[measurement][measure_signal],self._Matrix,Message[measurement][voxel_number])
        
        timer.cal_time()
        print(f"reconstrution time: ", timer.time[-1])
        timer.reset()    

    def _CGNR_origin(self,A,b,w):
        '''
        Attributes:
            -- A: m*n
            -- b: 1*m
            -- w: m*m

        Parameters:
            -- r: m*1
            -- z: n*1 (z_former and z_latter)
            -- p: n*1
            -- v: m*1

        Outputs:
            -- x: n*1   
        '''
        
        M = A.shape[0]
        N = A.shape[1]

        timer = Time()

        x = np.zeros(N, dtype = np.complex128)
        r = b.reshape(M)
        z_temp = np.transpose(np.conjugate(A)) @ w
        z_latter = z_temp @ r
        z_former = np.zeros(N,dtype = np.complex128)
        p = z_latter

        timer.cal_time()
        print(f"pre-work:", timer.time[-1])
        timer.begin_time()

        for i in tqdm(range(self._Iterations),desc="CGNR Reconstruction Calculation"):
            v = A @ p
            a = np.dot(np.conjugate(z_latter),z_latter) / (np.conjugate(v)@w@v + self._Lambda*(np.conjugate(p)@p))
            x += a * p
            r -= a * v
            z_former = z_latter
            z_latter = np.transpose(np.conjugate(A)) @ w @ r - self._Lambda * x
            beta = np.dot(np.conjugate(z_latter),z_latter) / np.dot(np.conjugate(z_former),z_former)
            p = z_latter + beta * p
        
        timer.cal_time()
        print(f"reconstrution time: ", timer.time[-1])
        timer.reset()

        return x

    def _ImageReshape(self,c,size):
        y = size[0]
        x = size[1]
        c = np.real(np.reshape(c,(y,x)))
        c /= np.max(c) 
        return c
    
    def _ImageRecon(self,A,b,w,size):
        origin_x = self._CGNR_origin(A,b,w)
        self._Image_data.append(self._ImageReshape(origin_x,size))

    def get_Image(self):
        return self._Image_data[0]    

import numpy as np

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
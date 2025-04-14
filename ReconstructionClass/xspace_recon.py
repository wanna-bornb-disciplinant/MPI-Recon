from email import message
from matplotlib.scale import scale_factory
import numpy as np
from tqdm import tqdm
from zmq import Message
from scipy.interpolate import griddata

from BaseClass.Constant_Base import *

from MPI_utils.weight_matrix import *
from MPI_utils.time import *
from MPI_utils.computation_with_timeout import *


class xspace_recon(object):
    def __init__(self,Message):
        super().__init__()

        self._Image_data = []
        self._ImageRecon(Message)
    
    def _ImageRecon(self,Message):
        ImagTan = self._Xspace_Recon(Message[measurement][measure_signal],Message[measurement][auxiliary_information].T)
        self._Image_data.append(self._Xspace_Reshape(Message[extend][pc_ffp].T,Message[extend][original_ffp].T,Message[extend][step_size],ImagTan))

    def _Xspace_Recon(self,U,ffp):

        # U = np.abs(U)
        U_norm = np.linalg.norm(U)
        scale_factor = 1 / U_norm
        U *= scale_factor
        # ffp = np.abs(ffp)
        # print(np.max(U),np.mean(U),np.min(U))

        temp = ffp ** 2
        ffpLen = np.sqrt(temp[0] + temp[1])
        ffpDir = np.divide(ffp, np.tile(ffpLen, (2, 1)))

        temp = U * ffpDir
        SigTan = temp[0] + temp[1]
        ImgTan = SigTan / ffpLen

        # print(np.max(ImgTan),np.mean(ImgTan),np.min(ImgTan))

        ImgTan = ImgTan / max(ImgTan[:])

        # print(np.max(ImgTan),np.mean(ImgTan),np.min(ImgTan))

        return ImgTan

    def _Xspace_Reshape(self,Rffp,ffp,step,ImagTan):
        pointx = np.arange(min(Rffp[0][:]), max(Rffp[0][:]) + step, step)
        pointy = np.arange(min(Rffp[1][:]), max(Rffp[1][:]) + step, step)
        xpos, ypos = np.meshgrid(pointx, pointy, indexing='xy')
        ImgTan = griddata((Rffp[0], Rffp[1]), ImagTan, (xpos, ypos), method='cubic')

        ImgTan = ImgTan[1:-1, 1:-1]
        ImgTan = ImgTan / np.max(ImgTan)
        return np.flip(ImgTan)

    def get_Image(self):
        return self._Image_data[-1]    

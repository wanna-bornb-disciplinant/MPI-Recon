import os
import urllib
import h5py
import numpy as np

from BaseClass.Reader_Base import *
from BaseClass.Constant_Base import *

class MDFReaderClass(Reader_Base):

    def __init__(self, SMNameFile, MeasNameFile):
        super().__init__()
        self._NameFileSM = SMNameFile
        self._NameFileMeas = MeasNameFile
        self._init_FileHandle()
        self._init_Message()

    # Get the file handle of the HDF5.
    def _init_FileHandle(self):
        self._MDF_SM = h5py.File(self._NameFileSM, 'r')
        self._MDF_Meas = h5py.File(self._NameFileMeas, 'r')

    def __get_SMData(self):
        S = self._MDF_SM[MDF_SM_data]
        return S[:, :, :, :] 

    def __get_MeasData(self):
        S = self.__MeasF[MDF_Meas_data]
        return S[:, :, :, :]

    def __get_BackGround(self):
        S = self.__SMF[ISBACKGROUNDFRAME]
        return S[:].view(bool)

    def __get_SamPointNum(self):
        S = self.__SMF[NUMSAMPLINGPOINTS]
        return int(np.array(S, dtype=np.int32))

    def __get_CaliSize(self):
        S = self.__SMF[CALIBRATIONSIZE]
        return S[:]

    # Initialize the Message.
    def _init_Message(self):

        self._set_MessageValue(MEASUREMENT, AUXSIGNAL, self.__get_SMData())
        self._set_MessageValue(MEASUREMENT, MEASIGNAL, self.__get_MeasData())
        self._set_MessageValue(MEASUREMENT, TYPE, SYSTEMMATRIX)
        self._set_MessageValue(MEASUREMENT, BGFLAG, self.__get_BackGround())
        self._set_MessageValue(SAMPLE, SAMNUMBER, self.__get_SamPointNum())
        self._set_MessageValue(MEASUREMENT, MEANUMBER, self.__get_CaliSize())

        return True

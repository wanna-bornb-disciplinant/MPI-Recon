import os
from abc import ABC, abstractmethod

from recon_final.BaseClass.Information_Base import *
from recon_final.BaseClass.Constant_Base import *

class Reader_Base(Information_Class,ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __get_FileHandle(self):
        pass

    def _send_2_information(self):
        pass





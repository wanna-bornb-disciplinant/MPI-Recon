import numpy as np

from abc import ABC,abstractmethod
from BaseClass.Constant_Base import *


class Phantom_Base(ABC):
    def __init__(self,diameter,saturation_mag_core,temperature,concentration): 
        '''调整量纲保证电压信号处在正常范围
            直径的单位:m(米)
            温度的单位:K(开尔文)
            玻尔兹曼常数:J/K(焦耳每开 1J = 1N*m)
            真空磁导率U0:U0=4pi*1e-7N/A^2=4pi*1e-7T*m/A 如果使用到4pi*1e-7,则最好不使用T/U0以免出现scale的错误
            核心的饱和磁化强度:T/U0 但实际上不会使用这个单位,在beta中需要和其他的一起计算,因此通常仍使用正常的磁场强度单位A/m,不使用T/U0
            根据上述量纲,通过朗之万等粒子响应模型计算出的

        '''
        self._diameter = diameter 
        self._temperature = temperature+T_BASE  #Kelvin Temperature
        self._particle_volume = self.__get_volume()
        self._saturation_mag_core = saturation_mag_core
        self._particle_magnetic_moment = self.__get_particle_magnetic()
        self._concentration_value = concentration     #concentration is a matrix
        self._beta_coeff = self.__get_beta_coeff()

        self._X = None
        self._Y = None

        self._Shape = None 
        self._Picture = None 
        
        
    #calculate the volume of particle
    def __get_volume(self):
        return PI*(self._diameter**3) / 6.0
    
    #calculate the magnetic moment of a single particle
    def __get_particle_magnetic(self):
        return self._particle_volume*self._saturation_mag_core

    #calculate the coeff of beta (Langevin Function)
    def __get_beta_coeff(self):
        return U0 * self._particle_magnetic_moment/(KB * self._temperature)
    
    #return the phantom matrix
    # def get_Phantom(self,x,y,concentration):
    #     self.get_Shape()
    #     return self._Picture
    
    @abstractmethod
    def get_Shape(self):
        pass

    @abstractmethod
    def get_Picture(self):
        pass    
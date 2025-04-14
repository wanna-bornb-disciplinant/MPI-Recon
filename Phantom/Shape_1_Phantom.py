import numpy as np 

from BaseClass.Phantom_Base import *

'''create a 'P' shape matrix to simulate the MPI reconstruction'''

class Phantom_Shape_1(Phantom_Base):
    '''
    Creating a phantom with complicated shape,seen in the simulation_path
    '''
    def __init__(self,temperature=20.0,diameter=30e-9,saturation_mag_core=8e5,concentration=5e7):
        super().__init__(diameter,saturation_mag_core,temperature,concentration)

    def get_Shape(self):  
        x, y = self._X, self._Y
        shape = np.zeros((y, x))

        center_1_x = int(x * 0.5)
        center_1_y = int(y * 0.2)
        radius = int(x*0.1)
        for x_in in range(x):
            for y_in in range(y):
                if (x_in - center_1_x)**2 + (y_in - center_1_y)**2 <= radius**2:
                    shape[y_in,x_in] = 0.25
        
        center_2_x = int(x * 0.2)
        center_2_y = int(y * 0.6)
        for x_in in range(x):
            for y_in in range(y):
                if (x_in - center_2_x)**2 + (y_in - center_2_y)**2 <= radius**2:
                    shape[y_in,x_in] = 0.5
        
        center_3_x = int(x * 0.8)
        center_3_y = int(y * 0.6)
        for x_in in range(x):
            for y_in in range(y):
                if (x_in - center_3_x)**2 + (y_in - center_3_y)**2 <= radius**2:
                    shape[y_in,x_in] = 1

        rect_height = int(x * 0.4)
        rect_width = int(x * 0.1)

        rect_center_1_x = int(x * 0.3)
        rect_center_1_y = int(y * 0.4)
        theta_1 = np.radians(-150)

        for x_in in range(x):
            for y_in in range(y):
                x_rot = (x_in - rect_center_1_x) * np.cos(theta_1) + (y_in - rect_center_1_y) * np.sin(theta_1)
                y_rot = -(x_in - rect_center_1_x) * np.sin(theta_1) + (y_in - rect_center_1_y) * np.cos(theta_1)

                if abs(x_rot) <= rect_height / 2 and abs(y_rot) <= rect_width / 2:
                    shape[y_in, x_in] = 1

        rect_center_2_x = int(x * 0.7)
        rect_center_2_y = int(y * 0.4)
        theta_2 = np.radians(-30)

        for x_in in range(x):
            for y_in in range(y):
                x_rot = (x_in - rect_center_2_x) * np.cos(theta_2) + (y_in - rect_center_2_y) * np.sin(theta_2)
                y_rot = -(x_in - rect_center_2_x) * np.sin(theta_2) + (y_in - rect_center_2_y) * np.cos(theta_2)

                if abs(x_rot) <= rect_height / 2 and abs(y_rot) <= rect_width / 2:
                    shape[y_in, x_in] = 0.5

        rect_center_3_x = int(x * 0.5)
        rect_center_3_y = int(y * 0.75)

        for x_in in range(x):
            for y_in in range(y):

                if abs(x_in - rect_center_3_x) <= rect_width / 2 and abs(y_in - rect_center_3_y) <= rect_height / 2:
                    shape[y_in, x_in] = 0.25
        
        return shape

    def get_Picture(self):
        # 可以矩阵Hadamard积形式计算，也可以广播形式计算
        return self._Shape * self._concentration_value
        
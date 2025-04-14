from platform import system
import numpy as np
import os
from tqdm import tqdm

from BaseClass.Simulation2D_Base import *
from BaseClass.Constant_Base import *
from MPI_utils.time import *

class SM_simulation_2D_Liss_FFP_Measurement(SM2D_SimulationBase):
    '''
    SM_simulation_2D_Liss_FFP_Measurement class and the instance is based on the SM2D_SimulationBase class
    SM2D_SimulationBase class define the basic parameters and the calculating voltage
    SM_simulation_2D_Liss_FFP_Measurement class will fill the abstract method, such as get_AuxSignal and translate_voltage
    The default is calculating the problems twice for Inverse Crime
    
    Parameters:
        -- Phantom "Phantom is the Simulation Image"
        -- isInverse "the class instance is calculating voltage and auxsignal in forward problem or reverse problem because of the Inverse Crime"
        -- Recal "whether to recalculate the system matrix"
        -- concentration_delta_volume "the calibration delta sample in measurement_based System Matrix Acquisition"
        -- noise_rate "the noise rate in the MPI reverse problem"
        -- Relative_Voltage_Path and Relative_AuxSignal_Path "if not Recal: need to know the voltage and auxsignal"
        -- Others "Attributes in Information_Base"
    '''
    def __init__(self,
                Phantom,
                isInverse,
                Recal = True,     
                SelectionField_X=2.0,  
                SelectionField_Y=2.0,
                DriveField_XA=12e-3,
                DriveField_YA=12e-3,
                DriveField_XF=2500000.0/102.0,
                DriveField_YF=2500000.0/96.0,
                DriveField_XP=PI/2.0, 
                DriveField_YP=PI/2.0, 
                Repeat_Time=6.528e-4,
                Sample_Frequency=2.5e6,
                concentration_delta_volume=1e-9, # unit volume is 1e-8, while delta volume should be further smaller
                Move_StepSize = 1e-4,
                noise_rate = 0.001,

                Relative_Voltage_Path = None,
                Relative_AuxSignal_Path = None,
                ):
        
        super(SM_simulation_2D_Liss_FFP_Measurement,self).__init__(
                Phantom,
                SelectionField_X,
                SelectionField_Y,
                DriveField_XA,
                DriveField_YA,
                DriveField_XF,
                DriveField_YF,
                DriveField_XP, 
                DriveField_YP, 
                Repeat_Time,
                Sample_Frequency,
                Move_StepSize,
                Recal,
                isInverse)
        
        if not self._isInverse:
            self._title_auxsignal = "Calculating AuxSignal in Forward Problem"
        else:
            self._title_auxsignal = "Calculating AuxSignal in Inverse Problem"

        #add noise
        self._noise_rate = noise_rate     

        #measurement-based approach
        self._delta_volume = concentration_delta_volume # 1e-9 < 1e-8
        self._delta_concentration = self._phantom._concentration_value
        
        if self._recal:
            self._send_2_information(topology_type="FFP",trajectory="Lissajous",wave_type_input="Sine",aux_type="SM based on measurement")          
            if self._isInverse:
                save_path2 = os.path.join(Result_Path,Relative_AuxSignal_Path)
                os.makedirs(os.path.dirname(save_path2),exist_ok=True)
                np.save(save_path2,self.message[measurement][auxiliary_information]) 
            else:
                save_path1 = os.path.join(Result_Path,Relative_Voltage_Path)
                os.makedirs(os.path.dirname(save_path1),exist_ok=True)
                np.save(save_path1,self.message[measurement][measure_signal])
        else:
            save_path1 = os.path.join(Result_Path,Relative_Voltage_Path)
            save_path2 = os.path.join(Result_Path,Relative_AuxSignal_Path)
            voltage_in = np.load(save_path1)
            aux_signal_in = np.load(save_path2)

            self._send_2_information(topology_type="FFP",trajectory="Lissajous",wave_type_input="Sine",aux_type="SM based on measurement",voltage=voltage_in,aux_signal=aux_signal_in)

    def __isOverScanRegion(self):
        pass

    def _add_noise_voltage(self,voltage):
        if self._noise_rate:
            noise_origin_x = np.random.normal(size = voltage[0,:].shape)
            noise_origin_y = np.random.normal(size = voltage[1,:].shape)
            noise_add_x = self._noise_rate * (np.sqrt(voltage[0,:]@voltage[0,:])) / (np.sqrt(noise_origin_x@noise_origin_x)) * noise_origin_x
            noise_add_y = self._noise_rate * (np.sqrt(voltage[1,:]@voltage[1,:])) / (np.sqrt(noise_origin_y@noise_origin_y)) * noise_origin_y
            voltage[0,:] += noise_add_x
            voltage[1,:] += noise_add_y
        return voltage


    def _translate_voltage(self,voltage):
        '''
           volatge shape is (2*sample_num)
           transform the voltage data to the frequency domain
        '''
        #DFT transform
        dft_x = np.fft.rfft(voltage[0,:]) # 1632/2+1 = 817
        dft_y = np.fft.rfft(voltage[1,:])
        dft_result =  np.concatenate((dft_x.reshape(1,-1),dft_y.reshape(1, -1)),axis = 1) # 1*m

        return dft_result           

    
    def _get_AuxSignal(self):
        
        system_matrix = np.zeros(((self._sample_num // 2 + 1) * 2,self._X_num * self._Y_num),dtype = complex)

        for i in tqdm(range(self._X_num * self._Y_num),desc=self._title_auxsignal):
            voltage = np.zeros((2,self._sample_num))
            voltage_frequency = np.zeros((1,(self._sample_num // 2 + 1) * 2))
            location_y = i // self._X_num
            location_x = i % self._X_num 
            for k in range(self._sample_num):
                '''no tqdm in inner loop'''
                coeff = (-1.0) * U0 * self._phantom._particle_magnetic_moment * self._phantom._beta_coeff * self._Drive_Derivative_sequence[k,:] * self._Sensiticity

                totalstrength_klocation = self._Drive_sequence[k,:] + self._selection_mag[location_y,location_x,:]
                total_mod = np.sqrt(np.add(totalstrength_klocation[0] ** 2,totalstrength_klocation[1] ** 2))

                Langevin_derivative = self.Langevin_derivative_2D(total_mod)

                voltage[0,k] = Langevin_derivative * coeff[0] * self._delta_volume * self._delta_concentration
                voltage[1,k] = Langevin_derivative * coeff[1] * self._delta_volume * self._delta_concentration

            voltage = self._add_noise_voltage(voltage)
            voltage_frequency = self._translate_voltage(voltage)

            for m in range((self._sample_num // 2 + 1) * 2):
                system_matrix[m,i] = voltage_frequency[0,m] / complex(self._delta_volume * self._delta_concentration,0)
        
        return system_matrix   



    




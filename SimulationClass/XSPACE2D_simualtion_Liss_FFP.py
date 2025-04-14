from platform import system
import numpy as np
import os
from tqdm import tqdm

from BaseClass.Simulation2D_Base import *
from BaseClass.Constant_Base import *
from MPI_utils.time import *

class Xspace_simulation_2D_Liss_FFP_Measurement(SM2D_SimulationBase):
    '''
    Xspace_simulation_2D_Liss_FFP_Measurement class and the instance is based on the SM2D_SimulationBase class
    SM2D_SimulationBase class define the basic parameters and the calculating voltage
    Xspace_simulation_2D_Liss_FFP_Measurement class will fill the abstract method, such as get_AuxSignal
    
    Parameters:
        -- Phantom "Phantom is the Simulation Image"
        -- concentration_delta_volume "the calibration delta sample in measurement_based System Matrix Acquisition"
        -- noise_rate "the noise rate in the MPI reverse problem"
        -- Relative_Voltage_Path and Relative_AuxSignal_Path "if not Recal: need to know the voltage and auxsignal"
        -- Others "Attributes in Information_Base"
    '''
    def __init__(self,
                Phantom,
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
                Move_StepSize = 1e-4,
                noise_rate = 0.001,
                Recal = True,
                Relative_Voltage_Path = None,
                ):
        
        super(Xspace_simulation_2D_Liss_FFP_Measurement,self).__init__(
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
                False)
        
        self._title_auxsignal = "Calculating AuxSignal in Forward Problem"

        #add noise
        self._noise_rate = noise_rate     
        
        if self._recal:
            self._send_2_information(topology_type="FFP",trajectory="Lissajous",wave_type_input="Sine",aux_type="Xspace based on measurement")          
            save_path1 = os.path.join(Result_Path,Relative_Voltage_Path)
            os.makedirs(os.path.dirname(save_path1),exist_ok=True)
            np.save(save_path1,self.message[measurement][measure_signal])
        else:
            save_path1 = os.path.join(Result_Path,Relative_Voltage_Path)
            voltage_in = np.load(save_path1)
            self._send_2_information(topology_type="FFP",trajectory="Lissajous",wave_type_input="Sine",aux_type="Xspace based on measurement",voltage=voltage_in)

    def __isOverScanRegion(self):
        pass

    def _translate_voltage(self,voltage):

        #DFT transform
        dft_x = np.fft.rfft(voltage[0,:]) # 1632/2+1 = 817
        dft_y = np.fft.rfft(voltage[1,:])
        dft_result =  np.concatenate((dft_x.reshape(1,-1),dft_y.reshape(1, -1)),axis = 1) # 1*m

        return dft_result   

    def _add_noise_voltage(self,voltage):
        if self._noise_rate:
            noise_origin_x = np.random.normal(size = voltage[0,:].shape)
            noise_origin_y = np.random.normal(size = voltage[1,:].shape)
            noise_add_x = self._noise_rate * (np.sqrt(voltage[0,:]@voltage[0,:])) / (np.sqrt(noise_origin_x@noise_origin_x)) * noise_origin_x
            noise_add_y = self._noise_rate * (np.sqrt(voltage[1,:]@voltage[1,:])) / (np.sqrt(noise_origin_y@noise_origin_y)) * noise_origin_y
            voltage[0,:] += noise_add_x
            voltage[1,:] += noise_add_y
        return voltage
       
    def _get_AuxSignal(self):
        Broad_selection_gradient = np.tile(self._selection_gradient,(1,self._sample_num)) # (2*M) 
        return np.divide(self._Drive_Derivative_sequence,Broad_selection_gradient.T) # (M*2)

    def _send_2_information(self,topology_type,trajectory,wave_type_input,aux_type,voltage=None):
        '''
        Parameters:
            -- topology_type: string  "select the FFP or the FFL"
            -- trajectory: string  "select the trajectory of the FFP, such as Lissajous Trajectory"
            -- wave_type_input: string  "select the waveform of the drive field, such as sine wave"
            -- aux_type: string  "select the method of the reconstruction, such as SM" 
        '''

        self._get_item1(particle_porperty,diameter,self._phantom._diameter)
        self._get_item1(particle_porperty,temperature,self._phantom._temperature)
        self._get_item1(particle_porperty,saturation_mag,self._phantom._saturation_mag_core)

        self._get_item1(selection_field,x_gradient,self._selection_x)
        self._get_item1(selection_field,y_gradient,self._selection_y)
        
        self._get_item2(drive_field,x_waveform,x_amplitude,self._drive_x_a)
        self._get_item2(drive_field,x_waveform,x_frequency,self._drive_x_f)
        self._get_item2(drive_field,x_waveform,x_phase,self._drive_x_p)
        self._get_item2(drive_field,y_waveform,y_amplitude,self._drive_y_a)
        self._get_item2(drive_field,y_waveform,y_frequency,self._drive_y_f)
        self._get_item2(drive_field,y_waveform,y_phase,self._drive_y_p)
        self._get_item1(drive_field,repeat_time,self._repeat_time) # self._repeat_time / k
        self._get_item1(drive_field,wave_type,wave_type_input)  
        
        '''no focus field'''

        self._get_item1(sample,topology,topology_type)
        self._get_item1(sample,frequency,self._sample_frequency)
        self._get_item1(sample,sample_number,self._sample_num)
        self._get_item1(sample,sample_time,self._repeat_time)
        self._get_item1(sample,sample_trajectory,trajectory)

        self._get_item1(measurement,recon_type,aux_type)
        self._get_item2(measurement,sensitivity,x_sensitivity,self._X_Sensiticity)
        self._get_item2(measurement,sensitivity,y_sensitivity,self._Y_Sensitivity)

        if self._recal:
            voltage_t = self._get_Voltage()
            # voltage_t = self._add_noise_voltage(voltage_t)
            self._get_item1(measurement,measure_signal,voltage_t)
        else:
            self._get_item1(measurement,measure_signal,voltage)
        
        aux_signal = self._get_AuxSignal()
        self._get_item1(measurement,auxiliary_information,aux_signal)
        self._get_item1(measurement,voxel_number,np.array([self._Y_num,self._X_num]))
        self._get_item1(measurement,voxel_size,self._voxel_size)

        self._get_item1(extend,original_ffp,self._FFPLocation)
        self._get_item1(extend,pc_ffp,self._FFPLocation_PC)
        self._get_item1(extend,step_size,self.move_step_size)
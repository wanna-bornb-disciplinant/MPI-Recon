import numpy as np

# dataloading and saving file

Result_Path = "recon_final\\result_path"
Simulation_Path = "recon_final\\simulation_path"
ExperimentV2_Path = "recon_final\\experiment_path"
Experiment_Path = "OpenMPI"

V2_SM_Path = "systemMatrix_V2.mdf"
V2_Mea_Path = "measurement_V2.mdf"

# constant list needed in the code

PI = np.pi
KB = 1.3806488e-23
T_BASE = 273.15
U0 = 4.0 * PI *1e-7
MOL = 6.022e23

EPS = 1e-11


# attribute in the information base

particle_porperty = 'Particle_Porperty'
diameter = 'Diameter'
temperature = 'Temperature'
saturation_mag = 'Saturation_Mag'

selection_field = 'Selection_Field'
x_gradient = 'X_Gradient'
y_gradient = 'Y_Gradient'
z_gradient = 'Z_Gradient'

drive_field = 'Drive_Field'
x_waveform = 'X_Waveform'
x_amplitude = 'X_Amplitude'
x_frequency = 'X_Frequency'
x_phase = 'X_Phase'
y_waveform = 'Y_Waveform'
y_amplitude = 'Y_Amplitude'
y_frequency = 'Y_Frequency'
y_phase = 'Y_Phase'
z_waveform = 'Z_Waveform'
z_amplitude = 'Z_Amplitude'
z_frequency = 'Z_Frequency'
z_phase = 'Z_Phase'
repeat_time = 'RepeatTime'
wave_type = 'WaveType'

focus_field = 'Focus_Field'
x_direction = 'X_Direction'
x_amplitude = 'X_Amplitude'
x_frequency = 'X_Frequency'
x_phase = 'X_Phase'
y_direction = 'Y_Direction'
y_amplitude = 'Y_Amplitude'
y_frequency = 'Y_Frequency'
y_phase = 'Y_Phase'
z_direction = 'Z_Direction'
z_amplitude = 'Z_Amplitude'
z_frequency = 'Z_Frequency'
z_phase = 'Z_Phase'
wave_type = 'WaveType'

sample = 'Sample'
topology = 'Topology'
sample_trajectory = 'Sample_Trajectory'
frequency = 'Frequency'
sample_number = 'Sample_Number'
sample_time = 'Sample_Time'

measurement = 'Measurement'
sensitivity = 'Sensitivity'
x_sensitivity = 'X_Sensitivity'
y_sensitivity = 'Y_Sensitivity'
z_sensitivity = 'Z_Sensitivity'
recon_type = 'Recon_Type'
measure_signal = 'Measure_Signal'
auxiliary_information = 'Auxiliary_Information'
voxel_number = 'Voxel_Number'
voxel_size = "Voxel_Size"

extend = 'Extend'
original_ffp = 'FFP_O'
pc_ffp = 'FFP_PC'
step_size = 'Step_Size'

# groups in the MDF data format
'''
    结合MDF论文的内容和实际MDF文件的目录结构来设置
'''
MDF_acquisition = "acquisition"
MDF_drivefield = "acquisition/drivefield"
MDF_receiver = "acquisition/receiver"
MDF_calibration = "calibration"
MDF_experiment = "experiment"
MDF_measurement = "measurement"
MDF_scanner = "scanner"
MDF_study = "study"
MDF_tracer = "tracer"

# important datasets in the MDF data format
MDF_data = "measurement/data"
MDF_voxel_number = "calibration/size"
MDF_CALI_FOV = "calibration/fieldOfView"
MDF_sample_num = "acquisition/receiver/numSamplingPoints"
MDF_isBackground = "/measurement/isBackgroundFrame"
MDF_bandwidth = "acquisition/receiver/bandwidth"

MDF_CALI_SNR = "calibration/snr"






import os 
import numpy
import sys
import h5py

from BaseClass.Constant_Base import *

SM_path1 = "OpenMPI/measurements/shapePhantom/1.mdf"
SM_path2 = "OpenMPI/measurements/shapePhantom/2.mdf"
SM_path3 = "OpenMPI/measurements/shapePhantom/3.mdf"

SM_path_con_path1 = "OpenMPI/measurements/concentrationPhantom/1.mdf"
SM_path_con_path2 = "OpenMPI/measurements/concentrationPhantom/2.mdf"
SM_path_con_path3 = "OpenMPI/measurements/concentrationPhantom/3.mdf"

SM_path_reso1 = "OpenMPI/measurements/resolutionPhantom/1.mdf"
SM_path_reso2 = "OpenMPI/measurements/resolutionPhantom/2.mdf"
SM_path_reso3 = "OpenMPI/measurements/resolutionPhantom/3.mdf"

SM_path_rota1 = "OpenMPI/measurements/rotationPhantom/1.mdf"
SM_path_rota2 = "OpenMPI/measurements/rotationPhantom/2.mdf"
SM_path_rota3 = "OpenMPI/measurements/rotationPhantom/3.mdf"
SM_path_rota4 = "OpenMPI/measurements/rotationPhantom/4.mdf"

Cali_path0 = "OpenMPI/calibrations/1.mdf"
Cali_path1 = "OpenMPI/calibrations/2.mdf"
Cali_path2 = "OpenMPI/calibrations/3.mdf"
Cali_path3 = "OpenMPI/calibrations/4.mdf"
Cali_path4 = "OpenMPI/calibrations/5.mdf"
Cali_path5 = "OpenMPI/calibrations/6.mdf"
Cali_path6 = "OpenMPI/calibrations/7.mdf"
Cali_path7 = "OpenMPI/calibrations/8.mdf"
Cali_path8 = "OpenMPI/calibrations/9.mdf"
Cali_path9 = "OpenMPI/calibrations/10.mdf"
Cali_path10 = "OpenMPI/calibrations/11.mdf"
Cali_path11 = "OpenMPI/calibrations/12.mdf"
Cali_path12 = "OpenMPI/calibrations/13.mdf"
Cali_path13 = "OpenMPI/calibrations/14.mdf"
Cali_path14 = "OpenMPI/calibrations/15.mdf"
Cali_path15 = "OpenMPI/calibrations/16.mdf"
Cali_path16 = "OpenMPI/calibrations/17.mdf"


# with h5py.File(SM_path, 'r') as f:
#     def print_datasets(name, obj):
#         if isinstance(obj, h5py.Dataset):
#             pass
#             #print(f"Dataset: {name}")
#         elif isinstance(obj, h5py.Group):
#             print(f"Group: {name}")

#     f.visititems(print_datasets)

# with h5py.File(Cali_path,"r") as f:
#     def print_datasets(name,obj):
#         if isinstance(obj, h5py.Dataset):pass
#             #print(f"Dataset: {name}")
#         elif isinstance(obj, h5py.Group):
#             print(f"Group: {name}")

#     f.visititems(print_datasets)

for i in range(17):
    file = eval(f"Cali_path{i}")
    print(h5py.File(file,"r")[MDF_data].shape,end = " ")
    print(h5py.File(file,"r")[MDF_sample_num][()],end = " ")
    print(h5py.File(file,"r")[MDF_voxel_number][()],end = " ")
    print(h5py.File(file,"r")[MDF_CALI_FOV][()],end = " ")
    print(h5py.File(file,"r")['/calibration/order'][()],end = " ")
    sum = 0
    is_background = h5py.File(file,"r")[MDF_isBackground][:].view(bool)
    for j in range(len(is_background)):
        if is_background[j] == True:
            sum += 1
    print(sum)

    print(np.where(is_background != 0)[0][0])

print("-"*10) 

for i in range(3):
    file = eval(f"SM_path{i+1}")
    print(h5py.File(file,"r")[MDF_data].shape,end = " ")
    print(h5py.File(file,"r")[MDF_sample_num][()],end = " ")
    sum = 0
    is_background = h5py.File(file,"r")[MDF_isBackground][:].view(bool)
    print(len(is_background),end=" ")
    for j in range(len(is_background)):
        if is_background[j] == True:
            sum += 1
    print(sum) #shape

print("-"*10)

for i in range(3):
    file = eval(f"SM_path_con_path{i+1}")
    print(h5py.File(file,"r")[MDF_data].shape,end = " ")
    print(h5py.File(file,"r")[MDF_sample_num][()],end = " ")
    sum = 0
    is_background = h5py.File(file,"r")[MDF_isBackground][:].view(bool)
    print(len(is_background),end=" ")
    for j in range(len(is_background)):
        if is_background[j] == True:
            sum += 1 
    print(sum) #concentration

print("-"*10)

for i in range(3):
    file = eval(f"SM_path_reso{i+1}")
    print(h5py.File(file,"r")[MDF_data].shape,end = " ")
    print(h5py.File(file,"r")[MDF_sample_num][()],end = " ")
    sum = 0
    is_background = h5py.File(file,"r")[MDF_isBackground][:].view(bool)
    print(len(is_background),end=" ")
    for j in range(len(is_background)):
        if is_background[j] == True:
            sum += 1 
    print(sum) #resolution

print("-"*10)

for i in range(4):
    file = eval(f"SM_path_rota{i+1}")
    print(h5py.File(file,"r")[MDF_data].shape,end = " ")
    print(h5py.File(file,"r")[MDF_sample_num][()],end = " ")
    sum = 0
    is_background = h5py.File(file,"r")[MDF_isBackground][:].view(bool)
    print(len(is_background),end=" ")
    for j in range(len(is_background)):
        if is_background[j] == True:
            sum += 1 
    print(sum) #rotation

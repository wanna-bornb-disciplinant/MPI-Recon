import matplotlib.pyplot as plt
from regex import T
from Phantom.P_Phantom import *
from Phantom.Shape_1_Phantom import Phantom_Shape_1

from BaseClass.Constant_Base import *
from SimulationClass.SM2D_simulation_Liss_FFP import *
from SimulationClass.XSPACE2D_simualtion_Liss_FFP import *
from MPI_utils.examine_2D_Simulation import *   
from ReconstructionClass.kaczmarz_recon import *
from ReconstructionClass.cgnr_recon import *
from ReconstructionClass.xspace_recon import *

import os



def Simulation_1(PhantomType, #phantom type(default 'P' shape)
                 relative_voltage_path,relative_auxsignal_path,noise_rate,
                 Recal,is_shuffle,reg_lambda,  #recalibration, this depends whether to calculate the simulation inverse
                 Temperature=20, Diameter=30e-9, Satur_Mag=8e5,Concentration=50, #50 mmol/L = 50 mol/m^3
                 Selection_X=2.0, Selection_Y=2.0,Drive_F_X=2500000.0/102.0,Drive_F_Y=2500000.0/96.0,Drive_A_X=12e-3,Drive_A_Y=12e-3,Drive_P_X=PI/2.0,Drive_P_Y=PI/2.0,
                 Sample_Time=6.528e-4,Sample_Frequency=2.5e6,
                 Delta_volume=1e-9,move_step_size=1e-4,inverse_step_size=1.2e-4,
                 ):
    '''
    2D Simulation based on System Matrix and Kaczmarz method
    
    Parameters:
        -- PhantomType: Choose the phantom type in simulation reconstruction (default: 0--Phantom Shape "P")
            1--Phantom Shape 1
        -- relative_voltage_path/relative_auxsignal_path: Combined with the result_path, it forms a storage path for voltage signals and system matrix signals
        -- Others: SM2D_simulation_Liss_FFP instance needed attributes
        -- inverse_step_size: the inverse crime need the inverse step

    Output:
        -- phantom: original image
        -- image1-7: 
            image1 denotes identity matrix, no Enforce_Rule and original Kaczmarz method
            image2 denotes energy matrix, no Enforce_Rule and Regularization Kaczmarz method
            image3 denotes normalization matrix, no Enforce_Rule and Rugularization Kaczmarz method
            image4 denotes learnable lambda,identity matrix, no Enforce_Rule and original Kaczmarz method
            image5 denotes identity matrix, Enforce_Rule and original Kaczmarz method
            image6 denotes learnable lambda, energy matrix, Enforce_Rule and Rugularization Kaczmarz method
            image7 denotes learnable lambda, normalization matrix, Enforce_Rule and Rugularization Kaczmarz method
    '''
    
    try:
        if PhantomType == 0:
            phantom = Phantom_Shape_P(temperature=Temperature, diameter=Diameter, saturation_mag_core=Satur_Mag, concentration=Concentration)
            # if inverse_concentration is not None:
            #     phantom_inverse = Phantom_Shape_P(temperature=Temperature, diameter=Diameter, saturation_mag_core=Satur_Mag, concentration=inverse_concentration)
        elif PhantomType == 1:
            phantom = Phantom_Shape_1(temperature=Temperature, diameter=Diameter, saturation_mag_core=Satur_Mag, concentration=Concentration)
        else:
            raise Exception("There is no such phantom type !!!")
    except:
        Exception("Phantom Module is broken !!!")
    
    try:
        simulation = SM_simulation_2D_Liss_FFP_Measurement(Phantom=phantom, 
                                                           Recal=Recal,
                                                           isInverse=False,
                                                           SelectionField_X=Selection_X, SelectionField_Y=Selection_Y,
                                                           DriveField_XA=Drive_A_X, DriveField_YA=Drive_A_Y, DriveField_XF=Drive_F_X, DriveField_YF=Drive_F_Y, DriveField_XP=Drive_P_X, DriveField_YP=Drive_P_Y,
                                                           Repeat_Time=Sample_Time, Sample_Frequency=Sample_Frequency,
                                                           concentration_delta_volume=Delta_volume,
                                                           Move_StepSize=move_step_size,
                                                           noise_rate=noise_rate,
                                                           Relative_Voltage_Path=relative_voltage_path, Relative_AuxSignal_Path=relative_auxsignal_path) 
    except:
        Exception("Simulation Module is broken !!!")

    simulation_inverse = SM_simulation_2D_Liss_FFP_Measurement(Phantom=phantom, 
                                                            Recal=Recal,
                                                            isInverse=True,
                                                            SelectionField_X=Selection_X, SelectionField_Y=Selection_Y,
                                                            DriveField_XA=Drive_A_X, DriveField_YA=Drive_A_Y, DriveField_XF=Drive_F_X, DriveField_YF=Drive_F_Y, DriveField_XP=Drive_P_X, DriveField_YP=Drive_P_Y,
                                                            Repeat_Time=Sample_Time, Sample_Frequency=Sample_Frequency,
                                                            concentration_delta_volume=Delta_volume,
                                                            Move_StepSize=inverse_step_size, #only one difference
                                                            noise_rate=noise_rate,
                                                            Relative_Voltage_Path=relative_voltage_path, Relative_AuxSignal_Path=relative_auxsignal_path)

    # the information of voltage and auxsignal is stored in the simulation instance and simulation_inverse instance
    if Recal:
        voltage_correct = np.load(os.path.join(Result_Path,relative_voltage_path))
        simulation_inverse._get_item1(measurement,measure_signal,voltage_correct)

    ImgData1 = SM_recon_Kaczmarz(simulation_inverse.message,Weight_type=1,is_Shuffle=is_shuffle,Recon_type=1,Lambda=reg_lambda) 
    result1 = ImgData1.get_Image() #无权重矩阵参与

    ImgData2 = SM_recon_Kaczmarz(simulation_inverse.message,Weight_type=1,is_Shuffle=is_shuffle,Recon_type=2,Lambda=reg_lambda)  
    result2 = ImgData2.get_Image()  #单位阵 

    ImgData3 = SM_recon_Kaczmarz(simulation_inverse.message,Weight_type=2,is_Shuffle=is_shuffle,Recon_type=2,Lambda=reg_lambda) 
    result3 = ImgData3.get_Image()  #能量阵

    ImgData4 = SM_recon_Kaczmarz(simulation_inverse.message,Weight_type=3,is_Shuffle=is_shuffle,Recon_type=2,Lambda=reg_lambda) 
    result4 = ImgData4.get_Image()  #归一化阵

    return phantom.get_Picture()/phantom._concentration_value, result1, result2, result3, result4



def Simulation_2(PhantomType, #phantom type(default 'P' shape)
                 relative_voltage_path,
                 Recal,  #recalibration, this depends whether to calculate the simulation inverse
                 Temperature=20, Diameter=30e-9, Satur_Mag=8e5,Concentration=0.1,
                 Selection_X=2.0, Selection_Y=2.0,Drive_F_X=2500000.0/102.0,Drive_F_Y=2500000.0/96.0,Drive_A_X=12e-3,Drive_A_Y=12e-3,Drive_P_X=PI/2.0,Drive_P_Y=PI/2.0,
                 Sample_Time=6.528e-4,Sample_Frequency=2.5e6,
                 move_step_size=1e-4,noise_rate=0.001
):
    '''
    2D Simulation based on X-Space
    
    Parameters:
        -- PhantomType: Choose the phantom type in simulation reconstruction (default: 0--Phantom Shape "P")
        -- relative_voltage_path/relative_auxsignal_path: Combined with the result_path, it forms a storage path for voltage signals and system matrix signals
        -- Others: SM2D_simulation_Liss_FFP instance needed attributes
        -- inverse_step_size: the inverse crime need the inverse step

    Output:
        -- phantom: original image
        -- image: 
            image1 denotes original X-space method image 
    '''
    try:
        if PhantomType == 0:
            phantom = Phantom_Shape_P(temperature=Temperature, diameter=Diameter, saturation_mag_core=Satur_Mag, concentration=Concentration)
        elif PhantomType == 1:
            phantom = Phantom_Shape_1(temperature=Temperature, diameter=Diameter, saturation_mag_core=Satur_Mag, concentration=Concentration)
        else:
            raise Exception("There is no such phantom type !!!")
    except:
        Exception("Phantom Module is broken !!!")
    
    # try:
    xspace_simulation = Xspace_simulation_2D_Liss_FFP_Measurement(Phantom=phantom, 
                                                        Recal=Recal,
                                                        SelectionField_X=Selection_X, SelectionField_Y=Selection_Y,
                                                        DriveField_XA=Drive_A_X, DriveField_YA=Drive_A_Y, DriveField_XF=Drive_F_X, DriveField_YF=Drive_F_Y, DriveField_XP=Drive_P_X, DriveField_YP=Drive_P_Y,
                                                        Repeat_Time=Sample_Time, Sample_Frequency=Sample_Frequency,
                                                        Move_StepSize=move_step_size,
                                                        noise_rate=noise_rate,
                                                        Relative_Voltage_Path=relative_voltage_path) 
    # except:
    #     Exception("Simulation Module is broken !!!")

    # the information of voltage and auxsignal is stored in the simulation instance and simulation_inverse instance
    
    image_data1 = xspace_recon(xspace_simulation.message)
    result1 = image_data1.get_Image()

    return phantom.get_Picture() / Concentration, result1

def Experiment_test():
    pass

if __name__ == "__main__":
    print("*" * 32)
    print("1: 2D MPI Simulation based on System Matrix and Kaczmarz method (default 'P' shape)")
    print("2: 2D MPI Simulation based on System Matrix and CGNR method (default 'P' shape)")
    print("3: 2D MPI experiment data reconstruction based on measurement_V2.mdf and systemMatrix_V2.mdf")
    print("4: xxxxxxx")
    print("5: xxxxxxx")
    print("6: xxxxxxx")
    print("7: xxxxxxx")
    print("8: xxxxxxx")
    print("Q: Quit")
    print("*" * 32)

    tutor  = input("select your own methods: ")
    default = ["1","2","Q"]

    while True:
        try:
            judge = tutor in default
            m = 1 / int(judge)
        except ZeroDivisionError:
            raise Exception("no such option !!!")        
        if tutor == "Q":
            break
        if tutor == "1":
            phantom,image1,image2,image3,image4 = Simulation_1(
                PhantomType=1,
                relative_voltage_path="shape_1/60dB_voltage.npy",relative_auxsignal_path="shape_1/60dB_auxsignal.npy",
                Recal=False,
                noise_rate=0.001,
                is_shuffle=False,
                reg_lambda=0.01            
            )
            #image1和image3不要
            phantom,image5 = Simulation_2(
                PhantomType=1,
                relative_voltage_path="shape_1/60dB_voltage_time.npy",
                Recal=False,
            )

            print(np.max(image5),np.mean(image5),np.min(image5))

            fig,axes = plt.subplots(nrows=2,ncols=4,figsize=(12,5))
            axes = axes.flatten()
            c1 = axes[0].imshow(phantom, cmap = "viridis")
            fig.colorbar(c1, ax = axes[0])
            axes[0].set_title("phantom",fontsize = 10)
            axes[0].axes.get_xaxis().set_visible(False)
            axes[0].axes.get_yaxis().set_visible(False)

            c2 = axes[1].imshow(image2, cmap = "viridis")
            fig.colorbar(c2, ax = axes[1])
            axes[1].set_title("Identity Recon",fontsize = 10)
            axes[1].axes.get_xaxis().set_visible(False)
            axes[1].axes.get_yaxis().set_visible(False)

            c3 = axes[2].imshow(image4, cmap = "viridis")
            fig.colorbar(c3, ax = axes[2])
            axes[2].set_title("Normalization Recon",fontsize = 10)
            axes[2].axes.get_xaxis().set_visible(False)
            axes[2].axes.get_yaxis().set_visible(False)

            # c4 = axes[3].imshow(image5, cmap = "viridis")
            # fig.colorbar(c4, ax = axes[3])
            # axes[3].set_title("X-space original",fontsize = 10)
            # axes[3].axes.get_xaxis().set_visible(False)
            # axes[3].axes.get_yaxis().set_visible(False)

            for i in range(3, len(axes)):
                axes[i].axis('off') 

            plt.tight_layout()
            fig_path = os.path.join(Result_Path,"shape_1/all_show.png")
            fig.savefig(fig_path)
            plt.show()
            break
        
        # if tutor == "2":
        #     phantom,image1,image2,image3 = Simulation_2(PhantomType=0,#Concentration=change_matrix,
        #         relative_voltage_path="P_shape/60dB_voltage.npy",relative_auxsignal_path="P_shape/60dB_auxsignal.npy",
        #         Recal=False,
        #         noise_rate=0.001)
        #     fig, axes = plt.subplots(nrows = 2,ncols = 4,figsize = (12,5))
        #     axes = axes.flatten()

        #     c1 = axes[0].imshow(phantom, cmap = "plasma", vmin = -0.2, vmax = 1.2)
        #     fig.colorbar(c1, ax = axes[0])
        #     axes[0].set_title("phantom",fontsize = 10)
        #     axes[0].axes.get_xaxis().set_visible(False)
        #     axes[0].axes.get_yaxis().set_visible(False)

        #     c2 = axes[1].imshow(image1, cmap = "plasma", vmin = -0.2, vmax = 1.2)
        #     fig.colorbar(c2, ax = axes[1])
        #     axes[1].set_title("energy_matrix CGNR",fontsize = 10)
        #     axes[1].axes.get_xaxis().set_visible(False)
        #     axes[1].axes.get_yaxis().set_visible(False)

        #     c3 = axes[2].imshow(image2, cmap = "plasma", vmin = -0.2, vmax = 1.2)
        #     fig.colorbar(c3, ax = axes[2])
        #     axes[2].set_title("identity_matrix CGNR",fontsize = 10)
        #     axes[2].axes.get_xaxis().set_visible(False)
        #     axes[2].axes.get_yaxis().set_visible(False)

        #     c4 = axes[3].imshow(image3, cmap = "plasma", vmin = -0.2, vmax = 1.2)
        #     fig.colorbar(c4, ax = axes[3])
        #     axes[3].set_title("norm_matrix CGNR",fontsize = 10)
        #     axes[3].axes.get_xaxis().set_visible(False)
        #     axes[3].axes.get_yaxis().set_visible(False)

        #     fig_path = os.path.join(Result_Path,"P_shape/60dB_recon_CGNR.png")
        #     fig.savefig(fig_path)
        #     plt.show()

        #     break


# co = axs[0, 1].contourf(X, Y, Z, levels=np.linspace(-1.25, 1.25, 11))
# fig.colorbar(co, ax=axs[0, 1])
# axs[0, 1].set_title('contourf()')



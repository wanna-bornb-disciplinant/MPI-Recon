# OpenMPIData文件简述

* 文件简述总结自"OpenMPIData: An initiative for freely accessible magnetic particle imaging data"一文和[https://magneticparticleimaging.github.io/OpenMPIData.jl/latest/index.html](https://magneticparticleimaging.github.io/OpenMPIData.jl/latest/index.html)这个关于OPenMPIData的官方说明文档

* 说明文档的引用文章cited as :
    + @article{knopp2020openmpidata,
    + title={OpenMPIData: An initiative for freely accessible magnetic particle imaging data},
    + author={Knopp, Tobias and Szwargulski, Patryk and Griese, Florian and Gr{\"a}ser, Matthias},
    + journal={Data in brief},
    + volume={28},
    + pages={104971},
    + year={2020},
    + publisher={Elsevier}
    + }
    
## 1
* 所有的OPenMPIData数据都可以在matlab,python,julia和C中被调用，数据也可以单独下载，下载链接是 [link](https://media.tuhh.de/ibi/openMPIData/data/)

* 文档中关于OPenMPIData的数据读取和重建调用，基于的编程语言都是科学计算使用的Julia，下面是OPenMPIData这个Julia包的使用方法：
```
using Pkg
Pkg.add(url="https://github.com/MagneticParticleImaging/OpenMPIData.jl.git")
```

* 在安装完包之后，可以使用包下载MPI数据：
```
using OpenMPIData
downloadOpenMPIData()
```

* 上述代码不会下载所有的OpenMPIData，高分辨率的MPI系统矩阵信息还需要额外的代码进行下载：
```
downloadCalibrationDataHighRes()
```

* 要下载1D、2D、3D和单独的系统矩阵信息，可以使用：
```
 download1DData()
 download2DData()
 download3DData()
 downloadCalibrationDataLowRes()
 downloadCalibrationDataHighRes()
```

## 2: Julia的重建示例
* 和数据集相同的，Julia的重建也放在了一个叫MPIReco.jl的包内，同时重建的过程中还需要PyPlot.jl的代码完成绘图操作：
```
using Pkg
Pkg.add(["MPIReco","PyPlot"])
```

* 3D激励场的重建示例如：
```
using OpenMPIData
include(joinpath(OpenMPIData.basedir(), "examples/reco3D.jl"))
```

* 上述重建代码的内容如下：
```
using PyPlot, MPIReco, OpenMPIData

include("visualization.jl")

filenameCalib = joinpath(OpenMPIData.basedir(),"data","calibrations","3.mdf")
#filenameCalib = joinpath(OpenMPIData.basedir(),"data","calibrations","6.mdf") # High Resolution

for (i,phantom) in enumerate(["shapePhantom", "resolutionPhantom", "concentrationPhantom"])

  filenameMeas = joinpath(OpenMPIData.basedir(),"data","measurements",phantom,"3.mdf")

  # reconstruct data
  c = reconstruction(filenameCalib, filenameMeas, iterations=3, lambd=0.001, bgCorrectionInternal=false,
                   minFreq=80e3, SNRThresh=2.0, recChannels=1:3, frames=1:1000, nAverages=1000)

  mkpath( joinpath(OpenMPIData.basedir(),"data/reconstructions/$(phantom)"))
  s = size(c)[2:4]
  
  # visualization
  if phantom =="shapePhantom"
    filenameImage = joinpath(OpenMPIData.basedir(),"data","reconstructions","$phantom","reconstruction3D.png")
    showMIPs(c[1,:,:,:,1],filename=filenameImage,fignum=i)
  elseif phantom =="resolutionPhantom"
    slice=[div(s[1]+1,2),div(s[2]+1,2),div(s[3]+1,2)]
    filenameImage = joinpath(OpenMPIData.basedir(),"data","reconstructions","$phantom","reconstruction3D.png")
    showSlices(c[1,:,:,:,1],slice,filename=filenameImage, fignum=i)
  elseif phantom =="concentrationPhantom"
    slice1 = [div(s[1],3)+1,div(s[2],3)+1,div(s[3],3)+1]
    slice2 = [2*div(s[1],3)+1,2*div(s[2],3)+1,2*div(s[3],3)+1]
    filenameImage = joinpath(OpenMPIData.basedir(),"data","reconstructions","$phantom","reconstruction3D_1.png")
    showSlices(c[1,:,:,:,1],slice1,filename=filenameImage,fignum=i)
    filenameImage = joinpath(OpenMPIData.basedir(),"data","reconstructions","$phantom","reconstruction3D_2.png")
    showSlices(c[1,:,:,:,1],slice2,filename=filenameImage,fignum=i+1)
  end
end
```
## 3: MPI Scanners
* OpenMPIData的相关数据都从下面的Bruker Preclinical MPI Scanner采集得到：
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/1.jpg" alt="1">
</div>

* 采集设备的相关参数如下：
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/2.jpg" alt="2">
</div>

## 4：Tracer
* MPI的成像高度依赖于使用的Tracer，OpenMPI中大多数数据集使用的Tracer都是perimag，其中有一个系统矩阵的数据使用的是Synomag-D

* **Perimag**是Micromod出品的一种tracer，通常可以与**Resovist**达到相同的MPI信号性能，对于可重复实验的研究至关重要

* **Synomag-D**也是由Micromod出品的一种造影剂，其成像性能大致是**Perimag**的两倍

## 5：Phantoms
* OpenMPIData中的所有数据都通过光刻3D打印得到，所有的phantom都使用相同的坐标系，坐标系中的X方向布置着机械装置，平面标记正的Z方向，根据右手定则确定Y方向

* phantom中的重建文件在.SLDPRT .STL .STEP中

* 第一类phantom是**Shape Phantom**,phantom中的重要参数包括：1mm半径尖端、10°顶角、22mm高度、总体积维683.9uL、使用50mmol/L的Perimag

* Shape Phantom可以以3D的锥体形式呈现，也可以在Y-Z平面以圆形式、X-Z或X-Y平面以具有扁平尖端的三角形形式呈现

* Shape Phantom的形象大致为：
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/3.jpg" alt="3">
</div>

* 第二类phantom是**Resolution Phantom**,这个phantom中包含五个使用50mmol/L的Perimag的管，5个管的一段有一个共同的原点，分别以不同的角度从X-Y平面和Y-Z平面延伸

* Y-Z平面的角度要小于X-Y的角度，原因是MPI系统在Z方向上有强磁场梯度，通过选择不同的平面，可以确定有距离不同而能够实现的分辨率

* Resolution Phantom的形象大致为：
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/4.jpg" alt="4">
</div>

* 第三类phantom是**Concentration Phantom**，phantom中包含8个2mm边长、8ul体积的立方，在X-Y平面中，两个立方的中心距离为12mm， 在Z方向上两个立方的中心距离为6mm

* 通常我们将8个立方体以这样的命名顺序将其称为1-8：顶层的前面、左边的立方体为1，顺时针排列1-4；底层的前面、左边的立方体为5，顺时针排列5-8

* 8个立方体的粒子浓度分别以1.5的比率依次减小，其大致表现为：
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/5.jpg" alt="5">
</div>

* 下图是8个立方体序号和浓度的对应关系：
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/6.jpg" alt="6">
</div>

* Concentration Phantom的形象大致为：
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/7.jpg" alt="7">
</div>

## 6：Measurement Sequences
* 在OpenMPIData中，通常采用三种不同的测量序列数据

* 第一种是1D序列，驱动场只在一维方向上发生变化，Y方向和Z方向上的移动由机械装置完成，先移动Z方向，再移动Y方向，表格中的一个Patch代表一个机械装置的移动位置，Y和Z方向共有361个位置
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/8.jpg" alt="8">
</div>

* 第二种是2D序列，驱动场在两个方向上发生变化，Z方向上由机械装置协助移动，共计19个位置
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/9.jpg" alt="9">
</div>

* 第三种是3D序列，驱动场在三个方向上发生变化
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/10.jpg" alt="10">
</div>

## 6：Calibration

* 如果需要使用一个基于系统矩阵的重建方法，OpenMPIData在**Calibration**部分提供了calibration data完成系统矩阵的组装，每一个calibration dataset都通过一个small sample进行测量，small sample会经过目标FOV的许多位置

* 即使是在1D和2D的序列情况下，OpenMPIData中也测量了3D的系统矩阵，这对于研究粒子物理特性是有帮助的，OpenMPIData建议在重建前选择合适的切片数据

* calibration datasets中还包括了FOV外的数据，**overscanning**标记了这些数据点，这对于排除掉FOV以外的数据对FOV以内的数据进行重建时的干扰，在数据重建前，可以手动地缩小overscan区域

* 1D、2D、3D的calibration data基本信息如下：
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/11.jpg" alt="11">
</div>
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/12.jpg" alt="12">
</div>
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/13.jpg" alt="13">
</div>

## 7：Datasets

* 所有的datasets都是在具体的phantom、MPI scanners、measurement sequences、tracer material的情况下得到

* 所有的phantom measurements如下：
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/14.jpg" alt="14">
</div>

* 所有的calibration measurements如下：
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/15.jpg" alt="15">
</div>

## 8：Reconstruction

* Shape Phantom的三种成像模式：1D、2D、3D的重建效果示例如下：
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/16.jpg" alt="16">
</div>
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/17.jpg" alt="17">
</div>
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/18.jpg" alt="18">
</div>

* Resolution Phantom的三种成像模式：1D、2D、3D的重建效果示例如下：
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/19.jpg" alt="19">
</div>
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/20.jpg" alt="20">
</div>
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/21.jpg" alt="21">
</div>

* Concentration Phantom的三种成像模式：1D、2D、3D的重建效果示例如下：
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/22.jpg" alt="22">
</div>
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/23.jpg" alt="23">
</div>
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/24.jpg" alt="24">
</div>
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/25.jpg" alt="25">
</div>
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/26.jpg" alt="26">
</div>
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/OpenMPIData_explanation/27.jpg" alt="27">
</div>

## 9:使用说明
* measurement中的MDF文件，以"OpenMPI/measurements/shapePhantom/1.mdf"为例，其排布为：

  + Group: acquisition
  + Group: acquisition/drivefield
  + Dataset: acquisition/drivefield/baseFrequency
  + Dataset: acquisition/drivefield/cycle
  + Dataset: acquisition/drivefield/divider
  + Dataset: acquisition/drivefield/numChannels
  + Dataset: acquisition/drivefield/phase
  + Dataset: acquisition/drivefield/strength
  + Dataset: acquisition/drivefield/waveform
  + Dataset: acquisition/gradient
  + Dataset: acquisition/numAverages
  + Dataset: acquisition/numFrames
  + Dataset: acquisition/numPeriods
  + Dataset: acquisition/offsetField
  + Group: acquisition/receiver
  + Dataset: acquisition/receiver/bandwidth
  + Dataset: acquisition/receiver/dataConversionFactor
  + Dataset: acquisition/receiver/numChannels
  + Dataset: acquisition/receiver/numSamplingPoints
  + Dataset: acquisition/receiver/unit
  + Dataset: acquisition/startTime
  + Group: experiment
  + Dataset: experiment/description
  + Dataset: experiment/isSimulation
  + Dataset: experiment/name
  + Dataset: experiment/number
  + Dataset: experiment/subject
  + Dataset: experiment/uuid
  + Group: measurement
  + Dataset: measurement/data
  + Dataset: measurement/isBackgroundCorrected
  + Dataset: measurement/isBackgroundFrame
  + Dataset: measurement/isFastFrameAxis
  + Dataset: measurement/isFourierTransformed
  + Dataset: measurement/isFramePermutation
  + Dataset: measurement/isFrequencySelection
  + Dataset: measurement/isSpectralLeakageCorrected
  + Dataset: measurement/isTransferFunctionCorrected
  + Group: scanner
  + Dataset: scanner/facility
  + Dataset: scanner/manufacturer
  + Dataset: scanner/name
  + Dataset: scanner/operator
  + Dataset: scanner/topology
  + Group: study
  + Dataset: study/description
  + Dataset: study/name
  + Dataset: study/number
  + Dataset: study/uuid
  + Dataset: time
  + Group: tracer
  + Dataset: tracer/batch
  + Dataset: tracer/concentration
  + Dataset: tracer/injectionTime
  + Dataset: tracer/name
  + Dataset: tracer/solute
  + Dataset: tracer/vendor
  + Dataset: tracer/volume
  + Dataset: uuid
  + Dataset: version

* calibration中的MDF文件，以"OpenMPI/calibrations/1.mdf"为例，其排布为：

  + Group: acquisition
  + Group: acquisition/drivefield
  + Dataset: acquisition/drivefield/baseFrequency
  + Dataset: acquisition/drivefield/cycle
  + Dataset: acquisition/drivefield/divider
  + Dataset: acquisition/drivefield/numChannels
  + Dataset: acquisition/drivefield/phase
  + Dataset: acquisition/drivefield/strength
  + Dataset: acquisition/drivefield/waveform
  + Dataset: acquisition/gradient
  + Dataset: acquisition/numAverages
  + Dataset: acquisition/numFrames
  + Dataset: acquisition/numPeriods
  + Dataset: acquisition/offsetField
  + Group: acquisition/receiver
  + Dataset: acquisition/receiver/bandwidth
  + Dataset: acquisition/receiver/dataConversionFactor
  + Dataset: acquisition/receiver/numChannels
  + Dataset: acquisition/receiver/numSamplingPoints
  + Dataset: acquisition/receiver/unit
  + Dataset: acquisition/startTime
  + Group: calibration
  + Dataset: calibration/fieldOfView
  + Dataset: calibration/fieldOfViewCenter
  + Dataset: calibration/method
  + Dataset: calibration/order
  + Dataset: calibration/size
  + Dataset: calibration/snr
  + Group: experiment
  + Dataset: experiment/description
  + Dataset: experiment/isSimulation
  + Dataset: experiment/name
  + Dataset: experiment/number
  + Dataset: experiment/subject
  + Dataset: experiment/uuid
  + Group: measurement
  + Dataset: measurement/data
  + Dataset: measurement/framePermutation
  + Dataset: measurement/isBackgroundCorrected
  + Dataset: measurement/isBackgroundFrame
  + Dataset: measurement/isFastFrameAxis
  + Dataset: measurement/isFourierTransformed
  + Dataset: measurement/isFramePermutation
  + Dataset: measurement/isFrequencySelection
  + Dataset: measurement/isSpectralLeakageCorrected
  + Dataset: measurement/isTransferFunctionCorrected
  + Group: scanner
  + Dataset: scanner/facility
  + Dataset: scanner/manufacturer
  + Dataset: scanner/name
  + Dataset: scanner/operator
  + Dataset: scanner/topology
  + Group: study
  + Dataset: study/description
  + Dataset: study/name
  + Dataset: study/number
  + Dataset: study/uuid
  + Dataset: time
  + Group: tracer
  + Dataset: tracer/batch
  + Dataset: tracer/concentration
  + Dataset: tracer/injectionTime
  + Dataset: tracer/name
  + Dataset: tracer/solute
  + Dataset: tracer/vendor
  + Dataset: tracer/volume
  + Dataset: uuid
  + Dataset: version

* 从group看，calibrations的MDF比measurement的要多一个calibration,都有的group包括acquisition、acquisition/drivefield、acquisition/receiver、experiment、study、measurement、scanner、tracer

* 从具体的datasets分析，calibration中多了这样几个datasets：calibration/fieldOfView、calibration/fieldOfViewCenter、calibration/method、calibration/order、calibration/size、calibration/snr，除此之外的区别在于calibration的MDF在measurement的group中多了一个measurement/framePermutation
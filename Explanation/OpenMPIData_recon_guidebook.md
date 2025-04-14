# OpenMPIData 重建相关信息简述

* 重建相关简述总结自[https://magneticparticleimaging.github.io/MPIReco.jl/dev/](https://magneticparticleimaging.github.io/MPIReco.jl/dev/)这个关于OPenMPIData的官方说明文档和对OPenMPI文件的分析、对多个现存的MPI重建软件的分析

## 1：指南的初衷

* 对MPI原始重建的MPIReco.jl的文档总结，也是对通过Julia使用OPenMPI数据的分析，具体有哪些参数可以用得上

## 2：MPIReco.jl的简介

* **MPIReco.jl**这个项目中包含了主要四个方向的函数实现：
    + 使用系统矩阵进行重建的基础重建算法
    + 使用focus field的MPI数据重建算法，即multi-patch的MPI重建
    + 多对比度的MPI重建
    + 针对系统矩阵的矩阵压缩技术

* MPIReco.jl的主要特征包括：
    + 频率过滤选择，只有在重建中使用到的频率范围才会存入内存进行计算
    + 依赖于**RegularizedLeastSquares.jl**实现的多种求解器
    + 使用者可以自行选择不同级别的MPI重建
    + 在**MPIFiles.jl**中实现了频谱泄露校正

## 3：安装

* ``add MPIReco``会安装MPIReco.jl和其依赖项RegularizedLeastSquares.jl和MPIFiles.jl

## 4：Getting Started

* 这里主要关心测试重建的代码的一些细节：
    + 首先重建的函数使用了Julia的一个特性： multiple dispatch mechanism（多重派发机制），类似重定义，可以根据参数的不同选择不同的函数实现
    + 除了测量数据和系统矩阵，重建函数还包括了以下参数：SNR阈值、重建帧、最小频率阈值、信号接收频道、迭代数、是否频谱泄露校正

## 5：基础重建

* 重建函数的多重派发机制根据函数标签进行不同级别的重建

### 5.1：on disk reconstruction

* 最高级别的重建是**on disk reconstruction**，函数标签为：
```
function reconstruction(d::MDFDatasetStore, study::Study,  
    exp::Experiment, recoParams::Dict)
```
* 这种重建适用于以study、exp等结构化存储的数据，重建的相关参数存储在recoParams中，对于已经计算过的特定的重建参数组合，直接会从硬盘中调取数据而不会重复计算

### 5.2：in memory reconstruction

* 第二个级别的重建时**in memory reconstruction**，函数标签为：
```
function reconstruction(recoParams::Dict)
```

* 这种重建方式可以通过不同的多重派发机制调用，但总的原则不变，一切重建的参数都存储在reconParams中，reconParams以三部分的形式组织：测量数据文件、系统矩阵文件、重建相关参数

### 5.3：middle level reconstruction

* 中等级别的重建首先区分数据集是multi-patch还是single-patch，对于两种重建各自有**reconstructionSinglePatch**和**reconstructionMultiPatch**，例如：
```
function reconstructionSinglePatch(bSF::MPIFile, bMeas::MPIFile;
                                  minFreq=0, maxFreq=1.25e6, SNRThresh=-1,
                                  maxMixingOrder=-1, numUsedFreqs=-1, sortBySNR=false, recChannels=1:numReceivers(bMeas),
                                  bEmpty = nothing, bgFrames = 1, fgFrames = 1,
                                  varMeanThresh = 0, minAmplification=2, kargs...)
```

* reconstructionSinglePatch实现了频率筛选，随后调用了下面的重建函数：
```
function reconstruction(bSF::MPIFile, bMeas::MPIFile, freq::Array;
  bEmpty = nothing, bgFrames = 1,  denoiseWeight = 0, redFactor = 0.0, thresh = nothing,
  loadasreal = false, solver = "Kaczmarz", sparseTrafo = nothing, saveTrafo=false,
  gridsize = gridSizeCommon(bSF), fov=calibFov(bSF), center=[0.0,0.0,0.0], useDFFoV=false,
  deadPixels=Int[], bgCorrectionInternal=false, kargs...)
```

* 上述的函数将重建的关于系统矩阵的信息调取后，将系统矩阵装配好，在调用更低一级的重建函数：
```
function reconstruction(S, bSF::MPIFile, bMeas::MPIFile, freq::Array, grid;
  frames = nothing, bEmpty = nothing, bgFrames = 1, nAverages = 1, numAverages=nAverages,
  sparseTrafo = nothing, loadasreal = false, maxload = 100, maskDFFOV=false,
  weightType=WeightingType.None, weightingLimit = 0, solver = "Kaczmarz",
  spectralCleaning=true, fgFrames=1:10, bgCorrectionInternal=false,
  noiseFreqThresh=0.0, kargs...)
```

### 5.4：low level reconstruction

* 低级别的重建函数为：
```
function reconstruction(S, u::Array; sparseTrafo = nothing,
                        lambd=0, progress=nothing, solver = "Kaczmarz",
                        weights=nothing, kargs...)
```

* 这里有两个地方要注意：第一个S可以增加一个维度，单独再存储一个系统矩阵转置的结果，方便在重建算法中使用；另外在这里可以使用数据压缩

## 6：重建结果

* MPIReco.jl的重建结果不仅包括重建数据，共计有5个维度，第一个维度表示**multi-spectral channels**，最后一个维度表示重建结果中的帧数**number of frames**(不是所有的重建数据都需要5个维度)


## 7：多对比度的MPI重建

* 仅需调用多个不同的系统矩阵文件：
```
bSFa = MPIFile(filenameA)
bSFb = MPIFile(filenameB)
```

* 调取数据通过**c[1,:,:,:]**和**c[2,:,:,:]**:
```
c = reconstruction([bSFa, bSFb], b;
                    SNRThresh=5, frames=1, minFreq=80e3,
                    recChannels=1:2, iterations=1)
```

## 8：multi-patch MPI重建

* 这里的重建函数调用比较灵活，例如：
```
bSFs = MultiMPIFile(["SF_MP01", "SF_MP02", "SF_MP03", "SF_MP04"])
mapping = [1,2,3,4]
freq = filterFrequencies(bSFs, SNRThresh=5, minFreq=80e3)
S = [getSF(SF,freq,nothing,"Kaczmarz", bgcorrection=false)[1] for SF in bSFs]
SFGridCenter = zeros(3,4)
FFPos = zeros(3,4)
FFPos[:,1] = [-0.008, 0.008, 0.0]
FFPos[:,2] = [-0.008, -0.008, 0.0]
FFPos[:,3] = [0.008, 0.008, 0.0]
FFPos[:,4] = [0.008, -0.008, 0.0]
c4 = reconstruction(bSFs, b; SNRThresh=5, frames=1, minFreq=80e3,
        recChannels=1:2,iterations=1, spectralLeakageCorrection=false,
        mapping=mapping, systemMatrices = S, SFGridCenter=SFGridCenter,
        FFPos=FFPos, FFPosSF=FFPos)
```

## 9：矩阵压缩技术

* MPIReco.jl中提供了两种矩阵压缩，在指定sparseTrafo之后，可以选择**DCT-IV**和**FFT**两种方式

* **radFactor**决定了矩阵的压缩率

## 10：MPIRF中的MPI实验数据重建

* MPIRF中实现了JSON文件的系统矩阵重建和X-Space重建，MDF文件的系统矩阵重建

### 10.1：MDF文件实现系统矩阵的具体实现

* 获取系统矩阵和测量数据的文件句柄，共计在文件读取的部分获取了两个文件中的五个数据：系统矩阵文件的/measurement/data，测量数据文件的/measurement/data，系统矩阵文件的/measurement/isBackgroundFrame(去除背景帧数据)，系统矩阵文件的/acquisition/receiver/numSamplingPoints,确定**w**和**v**即采样点数，系统矩阵文件的/calibration/size，即确定最终成像FOV的网格size

* 读取数据之后在重建之前经过一次重建前处理，前处理本身有一个基础的流程，MDF文件在此基础上继承。

* 基础的前处理代码中包含了验证数据格式，分别检查了磁粒子特性、梯度场信息、驱动场信息、focus field信息、采样相关信息、测量和重建所需信息的正确性，但都在一个前提下，就是数据本身不为None，这对应实验数据不一定能够获取完全的数据

* MDF文件的预处理部分没有进行标准化，只使用了数据裁切，首先使用 **/measurement/isBackgroundFrame**筛选了背景帧数据，背景帧的排列通常在所有帧的后面，用np.where筛选，之后使用rfft算出了频域的分量个数，通过DFT计算出了所有的频率分量的频率值，根据最开始设定的Threshold 80e3来筛选使用的最低频率值

* 这里根据两组数据来确定最后的操作，用于测试的V2数据分别有系统矩阵和测量数据两个文件，系统矩阵的data的维度是**1\*3\*817\*1959**这四个维度分别对应**J\*C\*K\*N**，J代表单帧中的周期数，C代表接收信号频道数，K代表频率分量数，N代表帧数,背景帧为23，对应**44\*44**的grid size；测量数据的data的维度是 **500\*1\*3\*1632**，对应于**N\*J\*C\*W**，不同的W代表的是空间位置的总数，因此W和K之间有**K = W/2+1**的关系

* 在OpenMPIData中举例，**download.jl**中将1D-data的resolution、shape、concentration三组数据的**1.mdf**放在一起，他们的数据维度都是**2\*36100\*3\*102**，2D-data的resolution、shape、concentration三组数据的**2.mdf**放在一起，他们的数据维度都是**2\*19000\*3\*1632**，3D-data的resolution、shape、concentration三组数据的**3.mdf**放在一起，他们的数据维度都是**2000\*1\*3\*53856**，低分辨率的calibrationdata是calibration中的1.mdf、2.mdf、3.mdf，各自对应一维、二维、三维的重建，data的数据维度分别是**1\*3\*52\*7221**、**1\*3\*817\*7221**，**1\*3\*26929\*7221**，高分辨率的calibrationdata是calibration中的4.mdf、5.mdf、6.mdf，各自对应一维、二维、三维的重建，data的数据维度分别是**1\*3\*52\*52023**、**1\*3\*817\*52023**，**1\*3\*26929\*52023**

* MPIRF剩下的前处理代码全是借鉴的，去掉了一个维度的信息(第三个接收信号的频道)，然后做维度合并(将系统矩阵合并成m*n型，测量信号合并成m*1型)，测量数据的多帧取平均值(合并测量信号的多帧)

* MPIRF关于系统矩阵的重建只实现了Kaczmarz算法，**lamda = numpy.linalg.norm(s,ord="fro")**

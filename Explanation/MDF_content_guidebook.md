# MDF格式数据文件简述

* 文件简述总结自"MDF: Magnetic Particle Imaging Data Format"一文

* 说明文档的引用文章cited as :
    + @article{knopp2016mdf, 
    + title={MDF: magnetic particle imaging data format}, 
    + author={Knopp, Tobias and Viereck, Thilo and Bringout, Gael and Ahlborg, Mandy and von Gladiss, Anselm and Kaethner, Christian and Neumann, Alexander and Vogel, Patrick and Rahmer, J{\"u}rgen and M{\"o}ddel, Martin},
    + journal={arXiv preprint arXiv:1602.06072},
    + year={2016}
    + }

## 1
* 系统矩阵的存储文件和测量数据的存储文件是分开的

## 2
* 单个MDF文件的group包括/study /experiment /scanner /acquisition /tracer /measurement /calibration /reconstruction

## 3
* /acquisition下面有两个subgroup: /acquisition/drivefield和/acquisition/receiver

## 4
* 按照第三点对每个group进行说明：
* **J**表示单个采集帧中包含的驱动场周期数
* **Y**表示单个驱动场周期中的梯度场的变化数，通常情况下梯度场是不会发生变化的，但有的时候梯度场的变化是为了采集不同精度下的数据
* **D**表示D个驱动场频道，即驱动场维度个数
* **F**表示单个频道下的驱动场信号频率数，即可能单个频道下的驱动场是多个不同频率的三角函数信号的总和
* **C**表示C个接收信号频道
* **A**表示多色MPI中tracer的个数
* **N**表示采集帧个数
* **O**表示N个采集帧中真正有粒子信号的帧数，即foreground帧数
* **E**表示N个采集帧中E个背景帧数，即没有粒子信号的帧数
* **B**表示稀疏表示中的稀疏个数
* **V**表示单个驱动场周期内采样点个数(单个接收频道内)
* **W**表示通过处理后的采样点个数，这些处理通常包括频率选择和带宽缩减等
* **K**表示经过rFFT之后的采样点个数，所以一般来说如果没有额外处理的情况下K = V/2+1
* 其他的常见的超参数都在/reconstruction中，Q表示的是重建数据集中的帧数，P表示重建数据中的体素个数,S表示重建数据中的频道数

## 5
* 按照第二条和第三条的顺序，给出所有的groups中的datasets：
* /
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/MDF_explanation/1.png" alt="1">
</div>

* /study
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/MDF_explanation/2.png" alt="2">
</div>

* /experiment
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/MDF_explanation/3.png" alt="3">
</div>

* /tracer
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/MDF_explanation/4.png" alt="4">
</div>

* /scanner
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/MDF_explanation/5.png" alt="5">
</div>

* /acquisition中的其他的dataset都很好理解，/gradient和/offsetField代表将梯度场变化理解为一个泰勒展开，一个中心位置的梯度场加上一个Jacobian矩阵
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/MDF_explanation/6.png" alt="6">
</div>
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/MDF_explanation/7.png" alt="7">
</div>

* /acquisition/drivefield中将单个频道内的驱动长周期看成是多个频率的叠加场
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/MDF_explanation/8.png" alt="8">
</div>
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/MDF_explanation/9.png" alt="9">
</div>

* /acquisition/receiver 在接收器group中，ADC采集信号和ADC实际测量得到的离散信号的转换关系为：
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/MDF_explanation/10.png" alt="10">
</div>
这里还有一个转换关系，就是在傅里叶变换之后，ADC的信号是经过一系列的增强和滤波得到的，因此感应线圈信号和ADC信号之间有一个转换因子：
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/MDF_explanation/11.png" alt="11">
</div>
inductionFactor中存储了MPS设备中的中心位置的感应电压强度，这个在MPI成像中不重要
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/MDF_explanation/12.png" alt="12">
</div>

* /measurement 不光是记录了数据本身，还记录了对数据进行的一系列处理，比如光谱泄露处理、背景校正、傅里叶变换、传递函数、频率选择、维度调整、帧顺序调整、前景信号压缩，操作除了压缩没什么好说的，压缩用两个公式说明，其实就是B和O决定了压缩率
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/MDF_explanation/13.png" alt="13">
</div>
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/MDF_explanation/14.png" alt="14">
</div>
当我压缩完要复原原来的数据时，我可以用上面的v来表示u，v其实就是把一些不重要的省略掉了，恢复时就直接拿v来取转换矩阵的共轭转置
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/MDF_explanation/15.png" alt="15">
</div>

* /calibration存储的都是校准系统的相关信息，offsetFields中存储的是MPS中的偏置场
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/MDF_explanation/16.png" alt="16">
</div>

* /reconstruction中有一个isOverscanRegion，Overscan指的是没有被扫描到的网格点
<div style="text-align: center;">
  <img src="https://gitee.com/ethanxiang0626/markdown_figs/raw/master/MDF_explanation/17.png" alt="17">
</div>

## 6
需要着重在Information_Base中提取得到的重建相关数据有：

* /acquisition中的/gradient /offsetField ***/numFrames*** ***/numPeriodsPerFrame***

* /acquisition/drivefield中的驱动场信息都可以存储，但 ***/numChannels*** 一定可以设置一个变量存储

* /acquisition/receiver中的 ***/numChannels*** 和 ***/numSamplingPoints***、**/bandwidth**也是必存的

* /measurement先确定data是否经过特殊处理，如果有特殊处理，可以选择读取其中的一些内容，但 ***/data*** 需要按照systemmatrix和测量数据进行展开存储 在MPIRF中额外存储了 ***/isBackgroundFrame*** ，希望说明对应的采集帧是否包含信号

* ***/data*** 有两个文件，一个是测量信号，一个是系统矩阵

* /calibration中存储的/deltaSampleSize /method /size /snr都是成像中需要注意的

* 在V2的例子里，在重建前进行了以下的处理：去除背景帧、squeeze()、根据bandwidth确定所有的频率分量的频，设置了一个阈值80e3来去除一些低频的信号、维度合并
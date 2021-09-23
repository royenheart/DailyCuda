# dailyCuda
每天一个Cuda练习

## Cuda安装

1. [进入官网查看显卡计算能力](https://developer.nvidia.com/zh-cn/cuda-gpus#compute)
2. [官网下载安装包](https://developer.nvidia.com/cuda-downloads)
3. 安装包根据流程进行下载，可以选择自定义安装去除不需要的[组件](#nsight系列)
4. 进入命令行终端键入`nvcc --version`查看是否安装完毕
   
## Nsight系列
[Nsight介绍文件](https://www.olcf.ornl.gov/wp-content/uploads/2020/02/Summit-Nsight-Systems-Introduction.pdf)

### Kernel Timeline
Cuda中主机代码调用核函数(**Kernel**,数据并行处理函数)进行GPU上的数据处理，一个Kernel对应一个Grid。

**Kernel Timeline**输出以Kernel为单位的特定时间的运行时间线，可以通过此观察GPU工作状态、Kernel运行时间等。

### Nsight Systems
nvprof的后继者，用于检测Kernel Timeline进行GPU，Cuda程序性能分析。

[下载地址](https://developer.nvidia.com/nsight-systems)

### Nsight Compute
[Nsight Systems](#nsight-systems)用于检测Kernel整体的运行，Nsight Compute可以提供单个Kernel内部的运行情况，比如Kernel的SASS汇编，运行时间等

[下载地址](https://developer.nvidia.com/nsight-compute)
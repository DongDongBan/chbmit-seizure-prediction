# 长程癫痫脑电信号常用处理工具

## 基本内容
<hr>
1. `chbmit_channels_meta.ipynb`用于提取CHB-MIT数据集每位患者的最大公共通道集合。出于尽可能保留更多公共通道的的目的，弃用了约10个EDF文件因其与其余文件完全不一致。
2. `chbmit_clean_retrieval.ipynb`利用上一阶段萃取的公共通道集合JSON文件，进行数据提取与通道映射，并转换原始summary文件为对计算机更友好的格式。
3. `raw_clean_mapping.json`前面两个流程的中间结果，测试正确性用
4. `segment_clean_dataset.ipynb`基于发作事件起止点和SOP-SPH-SAB-SAA四个超参数的时段划分代码
5. `utils`目录存储一些辅助函数和脚本
6. `plot`目录内有一些可视化代码
7. `features`目录内有一些经典时序特征提取代码和可选的高度优化性能的实现，可能依赖第三方库
8. `classification`目录存储了一些经典脑电和时序模型以清洗后数据集为输入的代码实现，可能包含高度优化性能的实现
9. `requirements.txt`指定了运行本代码仓库所需的最小环境

## 使用方法
<hr>

### 运行环境

#### 最小要求
#### 可选要求

### 代码组织

## TODO
<hr>
[] 编写一个通用性较高的DataLoader
[] 完成最基础的信号可视化代码
[] 在一些必要节点加一点测试代码
[] 整合进EHTZ, TUH数据集的处理代码
[] 仔细调研一下features需要使用哪些第三方库，怎么从机器学习/统计时序的角度组织子文件夹
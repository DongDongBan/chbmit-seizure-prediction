# 长程癫痫脑电信号常用处理工具

## 基本内容

1. [problem_definition.zh_CN.md](./problem_definition.zh_CN.md)问题定义描述了本仓库解决的中心任务
2. `create_label_tgt_classification.py`基于发作事件起止点和SOP-SPH-SAB-SAA四个超参数的时段划分代码
3. `utils`目录存储一些辅助函数和类
4. `plot`目录内有一些可视化代码
5. `features`目录内有一些经典时序特征提取代码和可选的高度优化性能的实现，可能依赖第三方库
6. `classification`目录存储了一些经典脑电和时序模型以清洗后数据集为输入的代码实现，可能包含高度优化性能的实现
7. `requirements.txt`指定了运行本代码仓库所需的最小环境

## 使用方法


### 运行环境

#### 最小要求

#### 可选要求

### 代码组织

## TODO

* [ ] 编写一个通用性较高的DataLoader
* [ ] 完成最基础的信号可视化代码
* [ ] 在一些必要节点加一点测试代码
* [ ] 整合进EHTZ, TUH数据集的处理代码
* [ ] 仔细调研一下features需要使用哪些第三方库，怎么从机器学习/统计时序的角度组织子文件夹

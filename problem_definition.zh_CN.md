每位患者的脑电数据集 $X$ 是包含 $k$ 个元素的序列 $X=\left( X_i \mid i=1,\cdots,k\right )$ ，其中第 $i$ 个元素 $X_i \in \mathbb{R}^{n \times m_i}$ 也称为第 $i$ 个区段，此处的 $n$ 表示通道数目而 $m_i$ 表示当前区段的采样次数。[^Xdim]这里假设 $X$ 的所有区段通道数目皆为 $n$ ，且其空间位置一致；所有区段都是等间隔采样，且所有区段的采样率相同皆为 $fs$ 。为了方便后续描述，我们引入符号 $\mathcal{T}_{X}$ 表示 $X$ 中包含的所有采样时刻 $dt$ 的集合。

[^Xdim]: 总而言之，序列 $X$ 有 $k$ 个元素，有 $\sum_{i=1}^{k}{m_i}$ 个样本，每个样本是一个 $n$ 维列向量。

![解释序列 X 和序列 O 的插图]()

我们将患者的发作期真实起止点也表示为一个二元组的序列 $O = \left( (O_{i\_start}, O_{i\_end}) \mid i=1,\cdots,q \right)$ ，其中一共有 $q$ 次发作，第 $i$ 发作的开始时刻和结束时刻分别对应 $O_{i\_start}$ 和 $O_{i\_end}$ ，这里假设所有 $O_{i\_start}$ 和 $O_{i\_end}$ 都属于 $\mathcal{T}_{X}$ 。

我们在这里引入两个辅助记号： $\overleftarrow{X}_{dt}$ 和 $\overleftarrow{O}_{dt}$ 分别表示序列 $X$ 和 $O$ 被时刻 $dt$ 截断且发生于 $dt$ 之前的部分。[^LArrow]

[^LArrow]: 这两个符号的严格定义 待完成 

基于此，不难得出理想化的发作估计任务目标： 在给定当前时刻 $dt_{cur}$ 和 $\overleftarrow{X}_{dt_{cur}}$ 时，尽可能准确地估计出 $O$ 的信息。既包括 $O$ 的长度 $q$ ，也涉及 $q$ 次发作的起止点 $O_{*\_start}$ 和 $O_{*\_end}$ 。
然而，基于现有技术水平要高精度实现这一目标十分困难，而且它要求估计了很多临床不实用的信息。因此，我们接下来逐步简化目标，依次分别提出了 _临床应用发作分期/预测目标_，_先验信息最大化前向分期/预测目标_ 和 _近似预测分期目标_。
为了方便理解这三个目标具体释义，我们再引入两个辅助记号： $O_{cur}$ 和 $O_{next}$ 。如果当前采样时刻 $dt_{cur}$ 处于某次发作中，我们用 $O_{cur}$ 表示这次发作区间；如果当前采样时刻 $dt_{cur}$ 不属于任何发作区间且其之后还有发作，则用 $O_{next}$ 表示其直接后继发作区间。

_临床应用发作分期/预测目标_ $Target_{clinical}$ 
:   鉴于临床实践中医生往往间隔一段时间才会对采集到的脑电记录进行集中标注，并且也只会 $O_{next}$ 采取干预措施。鉴于此，我们简化了需要估计的 $O$ 序列信息——已经标注的发作 $\overleftarrow{O}_{dt_{last}}$ 和 $O_{next}$ 之后的发作无需关注。
给定医生最后一次标注的采样时刻 $dt_{last}$ 和当前采样时刻 $dt_{cur}$ ，本目标需要估出的信息包括以下四方面：

    1. 检测当前时刻 $dt_{cur}$ 是否处于发作期，记为布尔变量 $B_{ost}$ 
    2. 如果 $B_{ost}$ 为真 ，则估计本次发作的区间 $[O_{cur\_start}, O_{cur\_end}]$ 
    3. 如果 $B_{ost}$ 为假，则估计下一次发作的起始点 $O_{next\_start}$ 
    4. 结合所有历史数据和已有标注估计 $dt_{last}$ 和 $dt_{cur}$ 之间的所有发作起止子序列 $\overline{O} = \left( (O_{j\_start}, O_{j\_end}) \mid j=l,\cdots,r \right)$ ，其中 $l$ 和 $r$ 分别表示紧跟 $dt_{last}$ 之后和紧邻 $dt_{cur}$ 之前的发作在 $O$ 中的索引，所有 $\overline{O}$ 中的发作区间都不包含 $dt_{last}$ 和 $dt_{cur}$ 本身

![解释三个目标 & 黑盒IO的插图]()

_先验信息最大化前向分期/预测目标_ $Target_{forward}$ 
:   由于过往发作信息的时效性远不如直接后继发作。因此，本目标在 $Target_{clinical}$ 的基础上不再区别 $dt_{last}$ 和 $dt_{cur}$ ，这就好比有一位自动化的标记员可以在每次采样后实时进行标注。因此，本目标需要估出的信息与 $Target_{clinical}$ 目标与 $Target_{clinical}$ 的相似，但所有的反向估计任务不再需要：

    1. 检测当前时刻 $dt_{cur}$ 是否处于发作期，记为布尔变量 $B_{ost}$ 
    2. 如果 $B_{ost}$ 为真 ，则估计本次发作的结束时刻 $O_{cur\_end}$ 
    3. 如果 $B_{ost}$ 为假，则估计下一次发作的起始点 $O_{next\_start}$ 

_近似预测分期目标_ $Target_{classification}$ 
:   为了进一步简化以在系统设计之初验证可行性，本目标被提出。它进一步简化 $Target_{forward}$ 目标中 $O_{next\_start}$ 回归问题为二分类问题，并不再要求估计 $O_{cur\_end}$ 。本目标需要将最新采样时刻 $dt_{cur}$ 划分为三个互斥的类别：

    1. 当前时刻 $dt_{cur}$ 处于发作期
    2. 当前时刻 $dt_{cur}$ 处于发作前期，即区间 $[dt_{cur}+SPH, dt_{cur}+SPH+SOP]$ 包含下一次发作起始点 $O_{next\_start}$ 
    3. 当前时刻 $dt_{cur}$ 处于发作间期，即区间 $[dt_{cur}+SPH, dt_{cur}+SPH+SOP]$ 不包含下一次发作起始点 $O_{next\_start}$ 
其中 SPH 和 SOP 是两个患者相关的超参数，分别用来描述患者的发作前期位置和长度。

[>SPH]: Seizure Prediction Horizon 用来描述发作预警至少应该给相关人员多长准备时间
[>SOP]: Seizure Onset Period 用来描述发作前期持续多长时间

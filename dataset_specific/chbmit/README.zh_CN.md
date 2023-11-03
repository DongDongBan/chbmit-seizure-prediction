1. `chbmit_channels_meta.ipynb`用于提取CHB-MIT数据集每位患者的最大公共通道集合。出于尽可能保留更多公共通道的的目的，弃用了约10个EDF文件因其与其余文件完全不一致。
1. `chbmit_clean_retrieval.ipynb`利用上一阶段萃取的公共通道集合JSON文件，进行数据提取与通道映射，并转换原始summary文件为对计算机更友好的格式。
1. `raw_clean_mapping.json`前面两个流程的中间结果，测试正确性用

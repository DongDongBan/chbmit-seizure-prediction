import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Subset, Dataset

from typing import Any, Iterator, Optional, Sequence, List, Sized, Union, Iterable, Callable, Tuple, Dict
# TODO 补充必要的doctest和完善不带外部数据依赖的UnitTest
# TODO 完成所有函数的 docstring
# TODO 小幅重构一下类型注解并尝试引入linter
#-------------------------- Utility Data Structure ----------------------------#
from enum import IntEnum
class Goal(IntEnum):
    CLASSIFICATION = 1, 
    REGRESSION = 2

from datetime import datetime
# from dataclasses import dataclass

# 辅助类：禁用数据Tensor的__setitem__方法防止被意外改写
# TODO 封装一个完全只有 const 操作的 TensorWrapper
class ReadOnlyTensorView(Tensor):
    def __init__(self, tensor: Tensor) -> None:
        super().__init__()
        self.data = tensor.data

    def __setitem__(self, key, value):
        raise RuntimeError("Cannot modify a read-only tensor.")

class TimeSeriesSeg:
    r"""Minimal DataClass to store an equal interval sampling ts record.
    
    Args:
        sample_rate: (int) 
        raw_data: (ReadOnlyTensorView) 2-dim instance of the Base Class or its Derived Class(e.g. torch.Tensor)
        start_dt: (datetime.datetime) Datetime at the very beginning of the record.
    """
    sample_rate: int
    raw_data: ReadOnlyTensorView
    start_dt: Optional[datetime]
    def __init__(self, sample_rate: int, raw_data: Union[Tensor, ReadOnlyTensorView], start_datetime: Optional[datetime]=None) -> None:
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError(f"sample_rate should be a positive integer value, but got sample_rate={sample_rate}")
        self.sample_rate = sample_rate
        if not isinstance(raw_data, (Tensor, ReadOnlyTensorView)) or len(raw_data.squeeze_().shape) != 2:
            try: 
                raw_data = torch.Tensor(raw_data).squeeze_()
                if len(raw_data.shape) != 2: raise AttributeError(f"Expected 2-dim raw_data, but got {len(raw_data.shape)}-dim")
            except Exception as exp:
                raise TypeError("raw_data argument should be convertible to a 2-dim torch.Tensor, but got "
                                f"{len(raw_data.shape)}-dim " if hasattr(raw_data, 'shape') else ""
                                f"raw_data={raw_data}") from exp
        self.raw_data = ReadOnlyTensorView(raw_data)
        # [Optional] start_datetime 对于某些简单的任务不是必需的
        if start_datetime is not None and not isinstance(start_datetime, datetime): 
            raise TypeError(f"start_datetime should be an instance of datetime.datetime, but got start_datetime={type(start_datetime)} {start_datetime}")
        self.start_dt = start_datetime

import copy
import warnings

class RegressionSeg(Dataset):
    r"""Map-Style subclass of torch.utils.data.Dataset for Remaining Time Estimation.
    
    Args: 
        sample_rate(int), X(Tensor-like), start_datetime(datetime.datetime): Used for construct self.segdata member. See :class:`TimeSeriesSeg`
        window_size(int)
        init_y(float): Value of the label at the very beginning of this seg.
    """
    segdata: TimeSeriesSeg
    _window_size: int # TODO 厘清 @property 在 type annotations
    task_type: Goal = Goal.REGRESSION
    X: ReadOnlyTensorView
    y_all: ReadOnlyTensorView
    y: Optional[ReadOnlyTensorView]
    _L: int 

    def __init__(self, X: ReadOnlyTensorView, init_y: float, window_size: int, sample_rate:int, start_datetime: Optional[datetime]=None, *args, **kwargs) -> None:
        self.segdata = TimeSeriesSeg(sample_rate, X, start_datetime)

        self.X = self.segdata.raw_data
        
        if not isinstance(init_y, float):
            try:
                init_y = float(init_y)
            except:
                raise TypeError(f"init_y should be convertible to float, but got {type(init_y)} init_y={init_y}")
        if init_y <= 0:
            warnings.warn(f"init_y should has a positive value, but got init_y={init_y}")            
        
        y_all = init_y - (torch.arange(start=0, end=self.X.shape[1], dtype=torch.double) / sample_rate)
        self.y_all = ReadOnlyTensorView(y_all)

        self.window_size = window_size # Would call self._update_label() internally to update self._L & self.y  

        super().__init__(*args, **kwargs)

    @property
    def window_size(self):
        return self._window_size
    @window_size.setter
    def window_size(self, val):
        if not isinstance(val, int) or val <= 0:
            raise ValueError(f"window_size should be a positive integer value, but got window_size={val}")
        if val > self.X.shape[1]:
            warnings.warn(f"window_size should be smaller than data length, got window_size={val}, data length={self.X.shape[1]}")
        self._window_size = val
        self._update_label()
    
    def __len__(self) -> int:
        return self._L if self._L > 0 else 0
    
    def __getattribute__(self, __name: str):
        if __name in ('raw_data', 'start_dt', 'sample_rate'):
            return self.segdata.__getattribute__(__name)
        return super().__getattribute__(__name)
    
    def __getitem__(self, index: int) -> Tuple[ReadOnlyTensorView, ReadOnlyTensorView]: # 由调用方保证不越界
        return self.X[:, index:(index+self.window_size)], self.y[index]
    
    # def __getitems__(self, indices: Sequence[int]) -> Tuple[ReadOnlyTensorView, ReadOnlyTensorView]: # 由调用方保证不越界
    #     raise NotImplementedError
    
    def _update_label(self) -> None:
        self._L = self.X.shape[1] - self.window_size + 1
        # Use the window's right-end side label as the whole window's label
        self.y = self.y_all[self.window_size-1:] if self._L > 0 else None
    
    def range_clone(self, start:Optional[int] = 0, end: Optional[int] = None) -> 'RegressionSeg': # 由调用方保证参数类型以及 start < end
        new_dst = copy.copy(self)
        new_dst.X = new_dst.segdata.raw_data[:, start:end]
        new_dst.y_all = new_dst.y_all[start:end]
        new_dst._update_label()

        return new_dst

class ClassificationSeg(Dataset):
    r"""Map-Style subclass of torch.utils.data.Dataset for TS Classification.
    
    Args: 
        sample_rate(int), X(Tensor-like), start_datetime(datetime.datetime): Used for construct self.segdata member. See :class:`TimeSeriesSeg`
        window_size(int)
        y(int): Value of the seg label.    
    """    
    segdata: TimeSeriesSeg
    _window_size: int # TODO 厘清 @property 在 type annotations
    task_type: Goal = Goal.CLASSIFICATION
    X: ReadOnlyTensorView
    y: Optional[int]
    _L: int 
    def __init__(self, X: ReadOnlyTensorView, y: int, window_size: int, sample_rate: int, start_datetime: Optional[datetime]=None, *args, **kwargs) -> None:
        self.segdata = TimeSeriesSeg(sample_rate, X, start_datetime)

        self.X = self.segdata.raw_data

        self.window_size = window_size # Would call self._update_label() internally to update self._L

        if not isinstance(y, int):
            warnings.warn(f"Param 'y' expected type int, but got type {type(y)}")
        self.y = y if self._L > 0 else None     

        super().__init__(*args, **kwargs)

    @property
    def window_size(self):
        return self._window_size
    @window_size.setter
    def window_size(self, val):
        if not isinstance(val, int) or val <= 0:
            raise ValueError(f"window_size should be a positive integer value, but got window_size={val}")
        if val > self.X.shape[1]:
            warnings.warn(f"window_size should be smaller than data length, got window_size={val}, data length={self.X.shape[1]}")
        self._window_size = val
        self._update_label()
            
    def __len__(self) -> int:
        return self._L if self._L > 0 else 0
    
    def __getattribute__(self, __name: str):
        if __name in ('raw_data', 'start_dt', 'sample_rate'):
            return self.segdata.__getattribute__(__name)
        return super().__getattribute__(__name)    
    
    def __getitem__(self, index: int) -> Tuple[ReadOnlyTensorView, int]: # 由调用方保证不越界
        return self.X[:, index:(index+self.window_size)], self.y

    # def __getitems__(self, indices: Sequence[int]) -> Tuple[ReadOnlyTensorView, ReadOnlyTensorView]: # 由调用方保证不越界
    #     raise NotImplementedError    
    
    def _update_label(self) -> None:
        self._L = self.X.shape[1] - self.window_size + 1

    def range_clone(self, start:Optional[int] = 0, end: Optional[int] = None): # 由调用方保证参数类型以及start < end
        new_dst = copy.copy(self)
        new_dst.X = new_dst.segdata.raw_data[:, start:end]
        new_dst._update_label()
        new_dst.y = new_dst.y if new_dst._L > 0 else None 
        return new_dst

#-------------------------- Core SplicedDataset ----------------------------#
import re
import bisect

EEGSeg = Union[ClassificationSeg, RegressionSeg]

class SplicedDataset(ConcatDataset):
    window_size: int
    sample_rate: int
    task_type: Goal
    # X_lst: List[TensorView] 
    # y_lst: List[Tenso|int]
    def __init__(self, datasets: Sequence[Dataset]) -> None: 
        def all_equal(iterable: Iterable[EEGSeg]):
            lst = list(iterable)
            return all(x == lst[0] for x in lst[1:])
        # datasets = list(datasets)

        if len(datasets) == 0: return None # ConcatDataset遇到空参数会报错，为了支持空参数这里特判一下
        if len(datasets) > 1:
            if not all_equal(map(lambda d: d.segdata.sample_rate, datasets)): # TODO d: Union[ClassificationSeg, RegressionSeg]
                raise ValueError("Inconsistent 'sample_rate' attributes of all datasets.")
            if not all_equal(map(lambda d: d.window_size, datasets)): # TODO d: Union[ClassificationSeg, RegressionSeg]
                raise ValueError("Inconsistent 'window_size' attributes of all datasets.") 
            if not all_equal(map(lambda d: d.task_type, datasets)): # TODO d: Union[ClassificationSeg, RegressionSeg]
                raise ValueError("Inconsistent 'task_type' attributes of all datasets.")   
            # TODO 用标准库组件替代 lambda 函数作 AttrGetter                           
        self.window_size = datasets[0].window_size
        self.sample_rate = datasets[0].segdata.sample_rate
        self.task_type = datasets[0].task_type

        super().__init__(datasets)

    # @property
    # def window_size(self) -> int:
    #     return self._window_size
    # @window_size.setter
    # def window_size(self, val: int) -> None:
    #     if not isinstance(val, int) or val <= 0:
    #         raise ValueError(f"window_size should be a positive integer value, but got window_size={val}") 
    #     # TODO 将下面的更改写成事务性可回滚的
    #     for seg in self.datasets:
    #         seg.window_size = val # 这样设计不好，因为.datasets元素皆有可能被多个split生成的实例共享，修改其window_size属性会影响别的实例，而别的实例又无法重新初始化
    #     self._window_size = val
    
    def rewsz(self, newsz: int) -> 'SplicedDataset':
        if not isinstance(newsz, int) or newsz <= 0:
            raise ValueError(f"window_size should be a positive integer value, but got window_size={newsz}")         
        newlst = [copy.copy(d) for d in self.datasets]
        for d in newlst: d.window_size = newsz
        return SplicedDataset(newlst)

    # TODO 慎重考虑下面三个split涉及的深浅拷贝问题
    def split_by_seg(self, seg_mask: str) -> Tuple['SplicedDataset', 'SplicedDataset', 'SplicedDataset']: # seg_mask: str 其中r表示训练，v表示验证，t表示测试， 其余字符表示忽略不用
        r"""
        
        """
        # 将一个dataset列表依据字符串表示的掩码划分成三个ConcatDataset并返回
        # 第三方库应该也有类似的实现吧，猜测
        assert(len(self.datasets) == len(seg_mask))
        # TODO assert(r_right < v_left && v_right < t_left)
        train_lst, valid_lst, test_lst = [], [], []
        for ds, ch in zip(self.datasets, seg_mask.lower()): # TODO split好像python又内置函数，之后改用
            if ch == 'r':
                train_lst.append(ds)
            elif ch == 'v':
                valid_lst.append(ds)
            elif ch == 't':
                test_lst.append(ds)
        return SplicedDataset(train_lst), SplicedDataset(valid_lst), SplicedDataset(test_lst)
    
    # 参数含义见 create_label_tgt_classification.py 中的 'Label' 字段
    # TODO 这个编码规则效率和简洁性还行，不过还是不太优雅
    def split_by_event(self, label_lst: List[str], event_mask: Dict[int, str], include_trailing_seg: str) -> Tuple['SplicedDataset', 'SplicedDataset', 'SplicedDataset']: # mask含义同split_by_seg， include_trailing_seg表示最后一次发作之后的尾随间期
        r"""
        
        """
        assert(len(self.datasets) == len(label_lst))
        # TODO assert(r_right < v_left && v_right < t_left)
        seg_mask: List[str] = []
        regex = r'[a-zA-Z]+([0-9+]+)-.+' # 这个与生成的seg label约定有关
        for label in label_lst: # for ds, label in zip(self, label_lst):
            match = re.search(regex, label)
            s = match.group(1)
            seg_mask.append(include_trailing_seg if s == '+' else event_mask[int(s)-1])
        return self.split_by_seg(seg_mask)
    
    # TODO 为这个方法编写更充分的测试
    def split_by_proportion(self, train_ratio: float=0.7, valid_ratio: float=0.1, *args, **kwargs) -> Tuple['SplicedDataset', 'SplicedDataset', 'SplicedDataset']: # 这个东西第三方库更好
        r"""
        
        """
        # 将一个ConcatDataset按比率分割，返回三个Subset
        total_points = sum(d.X.shape[1] for d in self.datasets)
        train_samples = round(train_ratio * total_points)
        valid_samples = round(valid_ratio * total_points)
        # test_samples = total_points - train_samples - valid_samples

        # 二分查找定位train-valid, valid-test分割点，不变式：
        # 这里使用bisect_left， bisect_right的不变式类似
        # bisect_left 返回的插入位置dataset_idx确保左侧一定严格小于，右侧大于等于
        
        # 左侧一定小于，保证该被一分为二的 self.datasets[dataset_idx] 的左半部分始终不能为空
        # 等于的情况可能出现在插入后右侧，此时 self.datasets[dataset_idx] 右半部分为空，全归左
        # 判定条件 idx == self.cumulative_sizes[dataset_idx] 

        t_v_dataset_idx = bisect.bisect_left(self.cumulative_sizes, train_samples)
        v_t_dataset_idx = bisect.bisect_left(self.cumulative_sizes, train_samples+valid_samples)

        # 特判 dataset_idx 为 0 的情况，计算seg内偏移量 sample_idx 
        if t_v_dataset_idx == 0: t_v_sample_idx = train_samples
        else: t_v_sample_idx = train_samples - self.cumulative_sizes[t_v_dataset_idx - 1]
        if v_t_dataset_idx == 0: v_t_sample_idx = train_samples+valid_samples
        else: v_t_sample_idx = train_samples+valid_samples - self.cumulative_sizes[v_t_dataset_idx - 1]

        # 开始分割工作，基于前面的分析，前一部分的尾半截一定存在，后一部分的头半截不一定存在
        train_dsts = self.datasets[:t_v_dataset_idx]
        train_dsts += [self.datasets[t_v_dataset_idx].range_clone(None, t_v_sample_idx)] 

        valid_dsts = [] if train_samples == self.cumulative_sizes[t_v_dataset_idx] \
                        else [self.datasets[t_v_dataset_idx].range_clone(t_v_sample_idx, None)] 
        valid_dsts += self.datasets[t_v_dataset_idx+1:v_t_dataset_idx]
        valid_dsts += [self.datasets[v_t_dataset_idx].range_clone(None, v_t_sample_idx)]

        test_dsts = [] if (train_samples+valid_samples) == self.cumulative_sizes[v_t_dataset_idx] \
                        else [self.datasets[v_t_dataset_idx].range_clone(v_t_sample_idx, None)]
        test_dsts += self.datasets[v_t_dataset_idx+1:]
        
        return SplicedDataset(train_dsts), SplicedDataset(valid_dsts), SplicedDataset(test_dsts)

# TODO 优雅解决类型标注中返回自身类型的情况   
SplitSpliced = Optional[SplicedDataset]

#-------------------------- Auxiliary Function ----------------------------#
# These functions are used to quickly create SplicedDataset class instances 
# by combining JSON generated from ../create_label_tgt_classification.py
class classification_label(IntEnum):
    inter = -1,
    pre = 0,
    onset = 1

import os, json
import numpy as np

# TODO 修改../create_label_tgt_classification.py 以支持 start_datetime 方便后期可视化
def get_classification_datasets(data_dir: str, json_path: str, window_size: int, sample_rate: int, \
                                label: IntEnum = classification_label) -> Tuple[List[EEGSeg], List[str]]:
    r'''
    
    '''
    with open(json_path) as f:
        seg_lst = json.load(f)    
    ds_lst = []; label_lst = []
    for seg in seg_lst:
        npy_path = os.path.join(data_dir, seg["File"])
        X = np.load(npy_path).squeeze()
        X = X[:, slice(*seg["Span"])]        
        if 'Pre' in seg["Label"]:
            ds_lst.append(ClassificationSeg(X, label.pre, window_size, sample_rate))            
        elif 'Inter' in seg["Label"]: 
            ds_lst.append(ClassificationSeg(X, label.inter, window_size, sample_rate))
        elif 'Onset' in seg["Label"]:
            ds_lst.append(ClassificationSeg(X, label.onset, window_size, sample_rate))
        else: 
            raise ValueError from seg["Label"]
        label_lst.append(seg["Label"])
    
    # return SplicedDataset(ds_lst), label_lst
    return ds_lst, label_lst

# TODO 修改../create_label_tgt_classification.py 以支持未知下次发作的Inter区段和 start_datetime
def get_regression_datasets(data_dir: str, json_path: str, window_size: int, sample_rate: int) \
                                -> Tuple[List[EEGSeg], List[str]]:
    r'''
    
    '''
    with open(json_path) as f:
        seg_lst = json.load(f)

    ds_lst = []; label_lst = []
    for seg in seg_lst:
        if 'Pre' in seg["Label"]:
            npy_path = os.path.join(data_dir, seg["File"])
            X = np.load(npy_path).squeeze()
            X = X[:, slice(*seg["Span"])]
            y0 = seg["PreSec"]; 
            ds_lst.append(RegressionSeg(X, y0, window_size, sample_rate))
            label_lst.append(seg["Label"])
        elif 'Inter' in seg["Label"]: 
            npy_path = os.path.join(data_dir, seg["File"])
            X = np.load(npy_path).squeeze()
            X = X[:, slice(*seg["Span"])]
            y0 = seg["PreSec"]; # 未知后续是否有发作的Inter区段 y0 设置成 str 而不是简单的 float 可能的一个值为 '7200.0+'
            if isinstance(y0, str) and y0.endswith('+'):
                y0 = float(y0[:-1])
            ds_lst.append(RegressionSeg(X, y0, window_size, sample_rate))
            label_lst.append(seg["Label"])
        else: 
            warnings.warn('The JSON file for a regression task should consist of a list of "seg" elements. In each "seg" element, '
                          'the "Label" field can startwith either "Pre" or "Inter" as valid substrings to indicate potential value. '
                          'Any other category substring should not be present.')
            # raise ValueError from seg["Label"]
        
    # return SplicedDataset(ds_lst), label_lst
    return ds_lst, label_lst

#-------------------------- CustomSampler ----------------------------#

r''' ![Deprecated] Due to the inefficiency of index generation and dereference when applied on ConcatDataset, this class has been deprecated
### 写成这种Sampler未必最优，因为每个返回的int还要再调用一次ConcatDataset的__getitem__，这是对数复杂度而非常量
### TODO 有空可以考虑直接写成一个带参迭代器得了，替代torch.utils.DataLoader这一层抽象！参考 DataLoader 源代码
class AugmentSequentialSampler(Sampler[int]):
    r"""Augmented SequentialSampler with two optional features: random_offset & step_size.
        Designed for online simulation and long-period data mining.

    Args:
        data_source (Dataset): dataset to sample from
        random_offset (bool): whether to drop some points at every sub-seg beginning.
        step_size(int|int (float)): integral value or function either indicating the step between two samples.
    """    
    # data_source: TimeSeriesSeg | SplicedDataset
    random_offset: bool
    step_size: Union[int, Callable[[float], int]]
    def __init__(self, data_source: Sized, random_offset: bool = True, step_size: Union[int, Callable[[float], int]] = 0) -> None:
        # TODO Add some checks
        self.data_source = data_source

        if not isinstance(random_offset, bool):
            raise TypeError(f"random_offset should be a boolean value, but got random_offset={random_offset}")        
        self.random_offset = random_offset

        if isinstance(step_size, int):
            self.step_size = step_size if step_size > 0 else data_source.window_size
        elif isinstance(step_size, type(lambda:0)): 
            self.step_size = step_size
        else:
            raise ValueError("step_size should be a positive integer value "
                             "or a function receives a float argument and returns an positive integer, "
                             f"but got step_size={step_size}")       

    def __iter__(self) -> Iterator[int]:
        window_size = self.data_source.window_size
        cumulative_sizes: List[int] = self.data_source.cumulative_sizes if isinstance(self.data_source, ConcatDataset) \
                                            else ConcatDataset.cumsum(self.data_source)
        start = 0; 
        for end in cumulative_sizes:
            start += int(torch.randint(high=window_size, size=(1,)).item()) if self.random_offset else 0
            while (start + window_size) < end:
                yield start
                # 根据step_size是整数还是函数分别计算
                start += self.step_size if isinstance(self.step_size, int) \
                                        else self.step_size(self.data_source[start][1]) # 以当前start对应时间窗的label值确定接下来的跨度
            start = end

    # Within step_size feature enabled, it's impossible to calculate __len__(self)
    # def __len__(self) -> int: 
'''

from collections import defaultdict
from torch.utils.data import DataLoader

class AugmentRandomDataLoader(DataLoader):
    r"""
    
    """
    class CatalogDataset(Dataset):
        def __init__(self, y_2_dst: Dict[int, ConcatDataset]) -> None:          
            super().__init__()
            concat_lst = list(y_2_dst.values())
            
            def infinite_random_generator(cdst: ConcatDataset):
                while True:
                    # 下面这句不知道为什么会卡死，看来迭代器和流式写法还是没学会
                    # yield from map(lambda k: cdst[k], torch.randperm(len(cdst), generator=self.generator))
                    for k in torch.randperm(len(cdst), generator=self.generator).tolist(): # 类内不放回抽样 replacement=False
                        yield cdst[k]  
            self.gen_lst = [infinite_random_generator(d) for d in concat_lst]

        def __getitem__(self, index: int) -> Tuple[ReadOnlyTensorView, int]: # TODO TA
            return next(self.gen_lst[index])
        # def __getitems__(self, indices: List[int]):
        #     pass
    
    # class WrappedInfWeightedSampler:
    #     raise NotImplementedError
    # TODO 支持无限长度 sampler wrapper 以及转发 generator 参数并区别开 Sampler & DataLoader 使用的生成器
    def __init__(self, ds_lst: Iterable[ClassificationSeg], ratio_dict: Dict[int, float], num_sample: int, *args, **kwargs) -> None:
        ds_lst = list(ds_lst)
        if not all(isinstance(ds, ClassificationSeg) for ds in ds_lst):
            raise TypeError(f"All elems in ds_lst arg should be an object of type ClassificationSeg")
        
        y_2_dst = AugmentRandomDataLoader.categorize_and_concat(ds_lst)
        inner_dst = AugmentRandomDataLoader.CatalogDataset(y_2_dst)
        weights_tensor = torch.tensor([ratio_dict[k] for k in y_2_dst.keys()], dtype=torch.double) # torch.tensor() will do input check by the way
        # 类别间放回抽样replacement=True，类内不放回抽样 replacement=False
        weighted_sampler = torch.utils.data.WeightedRandomSampler(weights_tensor, num_sample, replacement=True)
        super().__init__(inner_dst, *args, sampler=weighted_sampler, **kwargs)
    
    @staticmethod
    def categorize_and_concat(ds_lst: Iterable[ClassificationSeg]) -> Dict[int, ConcatDataset]:
        y_2_dst = defaultdict(list)
        for dst in ds_lst:
            y_2_dst[dst.y].append(dst) # categorize
        for k, v in y_2_dst.items():
            y_2_dst[k] = ConcatDataset(y_2_dst[k]) # Concat
        return y_2_dst

# TODO 多进程 SequentialDataLoader 建议先学习 NLP 和 Audio TS 的最佳实践再设计
def SequentialGenerator(data_source: Iterable[EEGSeg], random_offset: bool = True): # TODO TA
    r'''

    '''
    for idxable in data_source:       
        # No type check for idxable for simplicity
        idx = int(torch.randint(high=idxable.window_size, size=(1,)).item()) if random_offset else 0
        while idx < len(idxable):
            stride = (yield idxable[idx]) # Use the value sent through generator.send(v) as stride
            if not stride: idx += idxable.window_size
            else:          idx += stride



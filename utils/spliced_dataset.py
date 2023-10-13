import torch
from torch import Tensor
from torch.utils.data import ConcatDataset, Subset, Dataset

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union, Callable

#-------------------------- Utility Data Structure ----------------------------#
from enum import IntEnum
class Goal(IntEnum):
    CLASSIFICATION = 1, 
    REGRESSION = 2

from datetime import datetime
from dataclasses import dataclass

# 辅助类：禁用数据Tensor的__setitem__方法防止被意外改写
# TODO 考虑是否要禁用 __setattr__ / __setattribute__ 方法，否则 .fill_ 之类的带下划线原地方法还是可以改写
class ReadOnlyTensorView(Tensor):
    def __init__(self, tensor):
        super().__init__()
        self.data = tensor.data

    def __setitem__(self, key, value):
        raise RuntimeError("Cannot modify a read-only tensor.")

# TODO 补充 type annotations
# @dataclass
class TimeSeriesSeg:
    r"""Minimal DataClass to store an equal interval sampling ts record.
    
    Args:
        sample_rate: (int) 
        raw_data: (ReadOnlyTensorView) 2-dim instance of the Base Class or its Derived Class(e.g. torch.Tensor)
        start_dt: (datetime.datetime) Datetime at the very beginning of the record.
    """
    sample_rate: int
    raw_data: ReadOnlyTensorView # TODO Constrained to 2-dim Tensor
    start_dt: Optional[datetime]
    def __init__(self, sample_rate, raw_data, start_datetime: Optional[datetime]=None):
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError(f"sample_rate should be a positive integer value, but got sample_rate={sample_rate}")
        self.sample_rate = sample_rate
        if not isinstance(raw_data, (Tensor, ReadOnlyTensorView)) or raw_data.squeeze_().shape != 2:
            try: 
                raw_data = torch.Tensor(raw_data).squeeze_()
                if len(raw_data.shape) != 2: raise AttributeError(f"Expected 2-dim raw_data, but got {len(raw_data.shape)}-dim")
            except Exception as exp:
                raise TypeError(f"raw_data argument should be convertible to a 2-dim torch.Tensor, but got {len(raw_data.shape)}-dim raw_data={raw_data}") from exp
        self.raw_data = ReadOnlyTensorView(raw_data)
        # [Optional] start_datetime 对于某些简单的任务不是必需的
        if not isinstance(start_datetime, (datetime, None)):
            raise TypeError(f"start_datetime should be an instance of datetime.datetime, but got start_datetime={type(start_datetime)} {start_datetime}")
        self.start_dt = start_datetime

import copy
class RegressionSeg(Dataset):
    r"""Map-Style subclass of torch.utils.data.Dataset for Remaining Time Estimation.
    
    Args: 
        sample_rate(int), X(Tensor-like), start_datetime(datetime.datetime): Used for construct self.segdata member. See :class:`TimeSeriesSeg`
        window_size(int)
        init_y(float): Value of the label at the very beginning of this seg.
    """
    segdata: TimeSeriesSeg
    window_size: int
    task_type: Goal = Goal.REGRESSION
    X: ReadOnlyTensorView # TODO Constrained to 2-dim Tensor
    y: Tensor # TODO Constrained with typeof X
    _L: int 
    def __init__(self, X, init_y, window_size, sample_rate, start_datetime=None):
        self.segdata = TimeSeriesSeg(sample_rate, X, start_datetime)

        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(f"window_size should be a positive integer value, but got window_size={window_size}")
        self.window_size = window_size

        self.X = self.segdata.raw_data; self._update_L()
        self.y = init_y - (torch.arange(self._L, dtype=torch.double) / sample_rate) if self._L > 0 else None
        # self.y = ReadOnlyTensorView(self.y) # Use @ReadOnlyTensorView
    def __len__(self):
        return self._L if self._L > 0 else 0
    def __getitem__(self, index): # 由调用方保证不越界
        return self.X[:, index:(index+self.window_size)], self.y[index]
    def _update_L(self):
        self._L = self.X.shape[1] - self.window_size + 1
    def range_clone(self, start:Optional[int] = 0, end: Optional[int] = None): # 由调用方保证参数类型以及start < end
        new_dst = copy.copy(self)
        new_dst.X = new_dst.segdata.raw_data[:, start:end]
        new_dst._update_L()
        new_dst.y = new_dst.y[start:(end-self.window_size) if isinstance(end, int) else None] \
                    if new_dst._L > 0 else None

class ClassificationSeg(Dataset):
    r"""Map-Style subclass of torch.utils.data.Dataset for TS Classification.
    
    Args: 
        sample_rate(int), X(Tensor-like), start_datetime(datetime.datetime): Used for construct self.segdata member. See :class:`TimeSeriesSeg`
        window_size(int)
        y(int): Value of the seg label.    
    """    
    window_size: int
    task_type: Goal = Goal.CLASSIFICATION
    X: ReadOnlyTensorView # TODO Constrained to 2-dim Tensor
    y: int
    _L: int 
    def __init__(self, X, y, window_size, sample_rate, start_datetime=None):
        self.segdata = TimeSeriesSeg(sample_rate, X, start_datetime)

        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(f"window_size should be a positive integer value, but got window_size={window_size}")
        self.window_size = window_size

        self.X = self.segdata.raw_data; self._update_L(); self.y = y

    def __len__(self):
        return self._L if self._L > 0 else 0
    
    def __getitem__(self, index): # 由调用方保证不越界
        return self.X[:, index:(index+self.window_size)], self.y
    
    def _update_L(self):
        self._L = self.X.shape[1] - self.window_size + 1

    def range_clone(self, start:Optional[int] = 0, end: Optional[int] = None): # 由调用方保证参数类型以及start < end
        new_dst = copy.copy(self)
        new_dst.X = new_dst.segdata.raw_data[:, start:end]
        new_dst._update_L()
        new_dst.y = new_dst.y if new_dst._L > 0 else None 

#-------------------------- Core SplicedDataset ----------------------------#
import re
import bisect
# TODO 参照ConcatDataset以及Subset的源代码完成重构
class SplicedDataset(ConcatDataset):
    window_size: int
    sample_rate: int
    task_type: Goal
    # X_lst: List[TensorView] 
    # y_lst: List[Tenso|int]
    def __init__(self, datasets: Iterable[Dataset]): # Iterable[Dataset] 可能太宽松了
        # TODO 重写__getattr__方法将segdata子对象的成员暴露给Dataset对象
        def all_equal(iterable):
            return all(x == iterable[0] for x in iterable[1:])
        datasets = list(datasets)
        # assert len(self.datasets) > 0, 'datasets should not be an empty iterable' # Copied from ConcatDataset.__init__()
        if len(datasets) == 0: return None
        if len(datasets) > 1:
            if not all_equal(map(lambda d: d.segdata.sample_rate, datasets)): # TODO d: Union[ClassificationSeg, RegressionSeg]
                raise ValueError
            if not all_equal(map(lambda d: d.window_size, datasets)): # TODO d: Union[ClassificationSeg, RegressionSeg]
                raise ValueError 
            if not all_equal(map(lambda d: d.task_type, datasets)): # TODO d: Union[ClassificationSeg, RegressionSeg]
                raise ValueError                               
        self.window_size = datasets[0].window_size
        self.sample_rate = datasets[0].segdata.sample_rate
        self.task_type = datasets[0].task_type
        self.X_lst = list[map(lambda d: d.X.as_readonly_view(), datasets)] # TODO 将np.array 在这里原地转为 torch.const_tensorview
        self.y_lst = [d.y for d in datasets]
        super().__init__(datasets)

    # TODO 慎重考虑下面三个split涉及的深浅拷贝问题
    def split_by_seg(self, seg_mask): # seg_mask: str 其中r表示训练，v表示验证，t表示测试， 其余字符表示忽略不用
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
    
    # 参数含义见 get_classification_datasets
    def split_by_event(self, label_lst, event_mask, include_trailing_seg): # mask含义同split_by_seg， include_trailing_seg表示最后一次发作之后的尾随间期
        assert(len(self.datasets) == len(label_lst))
        # TODO assert(r_right < v_left && v_right < t_left)
        seg_mask = []
        regex = r'[a-zA-Z]+([0-9+]+)-.+' # 这个与生成的seg label约定有关
        for label in label_lst: # for ds, label in zip(self, label_lst):
            match = re.search(regex, label)
            s = match.group(1)
            seg_mask.append(include_trailing_seg if s == '+' else event_mask[int(s)-1])
        return self.split_by_seg(seg_mask)
    

    def split_by_proportion(self, train_ratio=0.7, valid_ratio=0.1): # 这个东西第三方库更好
        # 将一个ConcatDataset按比率分割，返回三个Subset
        total_points = sum(d.segdata.raw_data.shape[1] for d in self.datasets)
        train_samples = round(train_ratio * total_points)
        valid_samples = round(valid_ratio * total_points)
        # test_samples = total_points - train_samples - valid_samples

        # 二分查找定位train-valid, valid-test分割点，不变式为
        t_v_dataset_idx = bisect.bisect_left(self.cumulative_sizes, train_samples)
        v_t_dataset_idx = bisect.bisect_left(self.cumulative_sizes, train_samples+valid_samples)
        if t_v_dataset_idx == 0: t_v_sample_idx = train_samples
        else: t_v_sample_idx = train_samples - self.cumulative_sizes[t_v_dataset_idx - 1]
        if v_t_dataset_idx == 0: v_t_sample_idx = train_samples+valid_samples
        else: v_t_sample_idx = train_samples+valid_samples - self.cumulative_sizes[v_t_dataset_idx - 1]

        # 开始分割工作
        train_dsts = self.datasets[:t_v_dataset_idx]
        train_dsts += [self.datasets[t_v_dataset_idx].range_clone(None, t_v_sample_idx)]
        valid_dsts = [] if train_samples == self.cumulative_sizes[v_t_dataset_idx] \
                        else [self.datasets[t_v_dataset_idx].range_clone(t_v_sample_idx, None)] 
        valid_dsts += self.datasets[t_v_dataset_idx+1:v_t_dataset_idx]
        valid_dsts += [self.datasets[v_t_dataset_idx].range_clone(None, v_t_sample_idx)]
        test_dsts = [] if (train_samples+valid_samples) == self.cumulative_sizes[v_t_dataset_idx] \
                        else [self.datasets[v_t_dataset_idx].range_clone(v_t_sample_idx, None)]
        test_dsts += self.datasets[v_t_dataset_idx+1:]
        
        return SplicedDataset(train_dsts), SplicedDataset(valid_dsts), SplicedDataset(test_dsts)

#-------------------------- Auxiliary Function ----------------------------#
# These functions are used to quickly create SplicedDataset class instances 
# by combining JSON generated from ../segment_clean_dataset.ipynb
class CLASSIFICATION_LABEL(IntEnum):
    INTER = -1,
    PRE = 0,
    ONSET = 1

import os, json
import numpy as np
def get_classification_datasets(data_dir, json_path, window_size, sample_rate, label = CLASSIFICATION_LABEL):
    with open(json_path) as f:
        seg_lst = json.load(f)    
    ds_lst = []; label_lst = []
    for seg in seg_lst:
        npy_path = os.path.join(data_dir, seg["File"])
        X = np.load(npy_path).squeeze()
        X = X[:, slice(*seg["Span"])]        
        if 'Pre' in seg["Label"]:
            ds_lst.append(ClassificationSeg(X, label.PRE, window_size))            
        elif 'Inter' in seg["Label"]: 
            ds_lst.append(ClassificationSeg(X, label.INTER, window_size))
        elif 'Onset' in seg["Label"]:
            ds_lst.append(ClassificationSeg(X, label.ONSET, window_size))
        else: 
            raise ValueError from seg["Label"]
        label_lst.append(seg["Label"])
    return SplicedDataset(zip(ds_lst, label_lst))



def get_regression_datasets(data_dir, json_path, window_size, sample_rate, clamp):
    with open(json_path) as f:
        seg_lst = json.load(f)
    
    # if not clamp: # 本来想支持自动检测钳位值的，但是异常处理太麻烦了 # TODO
    #     clamp = 0
    #     for seg in seg_lst: 
    #         if "PreSec" in seg:
    #             clamp = max(clamp, seg["PreSec"])
    # assert(clamp > 0)

    ds_lst = []; label_lst = []
    for seg in seg_lst:
        if 'Pre' in seg["Label"]:
            npy_path = os.path.join(data_dir, seg["File"])
            X = np.load(npy_path).squeeze()
            X = X[:, slice(*seg["Span"])]
            y0 = seg["PreSec"]; assert(y0 <= clamp)
            ds_lst.append(RegressionSeg(X, y0, window_size, sample_rate))
            label_lst.append(seg["Label"])
        elif 'Inter' in seg["Label"]: # 默认所有inter都超过限幅clamp，不然直接用get_simulation_dataloader更方便！
            npy_path = os.path.join(data_dir, seg["File"])
            X = np.load(npy_path).squeeze()
            X = X[:, slice(*seg["Span"])]
            y0 = seg["PreSec"]; assert(y0 > clamp or y0 == -1) # Debug
            ds_lst.append(ClassificationSeg(X, clamp, window_size))
            label_lst.append(seg["Label"])
        else: # 暂时不支持混合发作期进来 # TODO
            pass # raise ValueError from seg["Label"]
        
    return SplicedDataset(zip(ds_lst, label_lst))

#-------------------------- CustomSampler ----------------------------#
from torch.utils.data import RandomSampler, Sampler
from collections import defaultdict

class AugmentRandomSampler(Sampler[int]):
    r"""Samples elements randomly from SplicedDataset instance using specified category-weights. 

    Args:
        data_source (SplicedDataset): dataset to sample from
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """
    data_source: SplicedDataset
    weights: Tensor
    num_samples: int

    def __init__(self, data_source: SplicedDataset, weights: Sequence[float],
                 num_samples: int, generator=None) -> None:
        if not isinstance(data_source, SplicedDataset) or data_source.task_type != Goal.CLASSIFICATION:
            raise ValueError("data_source should be an instance of class SplicedDataset within task_type set to Goal.CLASSIFICATION, "
                             f"but got data_source={data_source}")
        self.data_source = data_source 

        self.generator = generator

        # TODO 将weights_tensor改为类型Dict[int, double]
        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        if len(weights_tensor.shape) != 1:
            raise ValueError("weights should be a 1d sequence but given "
                             f"weights have shape {tuple(weights_tensor.shape)}")        
        self.weights = weights_tensor

        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={num_samples}")
        assert(num_samples <= len(data_source)) # TODO 暂时不支持这种情况，除非已知 torch.multinomial 支持
        self.num_samples = num_samples
   
    def __iter__(self) -> Iterator[int]:
        sz_counter = defaultdict(int)
        for dst in self.data_source.datasets:
            sz_counter[dst.y] += len(dst)
        
        final_weights = [Tensor(size=(len(dst),), dtype=torch.double).fill_(self.weights[dst.y] / sz_counter[dst.y]) 
                         for dst in self.data_source.datasets]
        final_weights = torch.concat(final_weights)
        rand_tensor = torch.multinomial(final_weights, self.num_samples, False, # Use False instead of removed member self.replacement, 
                                        generator=self.generator)
        yield from iter(rand_tensor.tolist())        

    def __len__(self) -> int:
        return self.num_samples

### 写成这种Sampler未必最优，因为每个返回的int还要再调用一次ConcatDataset的__getitem__，这是对数复杂度而非常量
### 有空可以考虑直接写成一个带参迭代器得了，替代torch.utils.DataLoader这一层抽象！
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
            self.step_size = step_size if step_size != 0 else data_source.window_size
        # TODO elif isinstance(step_size, int (float)): 
        #   self.step_size = step_size
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

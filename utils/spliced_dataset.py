import torch
from torch import Tensor

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union, Callable

from enum import IntEnum

class Goal(IntEnum):
    CLASSIFICATION = 1, 
    REGRESSION = 2

from torch.utils.data import ConcatDataset, Subset, Dataset
import re
# TODO 参照ConcatDataset以及Subset的源代码完成重构
class SplicedDataset(ConcatDataset):
    window_size: int
    sample_rate: int
    task_type: Goal
    X_lst: NotImplemented
    y_lst: List[Tensor]
    def __init__(self, subsegs):
        # self.window_size = window_size
        # self.sample_rate = sample_rate
        # self.task_type = Goal.REGRESSION
        # self.X_lst = list[map(lambda x: x[0].as_readonly_view, subsegs)] # TODO 将np.array 在这里原地转为 torch.const_tensorview
        # self.y_lst = [generate_y(x.shape[1], task_type) for x in subsegs]
        pass
        super().__init__(*subsegs)

    def split_by_seg(self, seg_mask): # seg_mask: str 其中r表示训练，v表示验证，t表示测试， 其余字符表示忽略不用
        # 将一个dataset列表依据字符串表示的掩码划分成三个ConcatDataset并返回
        # 第三方库应该也有
        assert(len(self) == len(seg_mask))
        # TODO assert(r_right < v_left && v_right < t_left)
        train_lst, valid_lst, test_lst = [], [], []
        for ds, ch in zip(self, seg_mask.lower()): # TODO split好像python又内置函数，之后改用
            if ch == 'r':
                train_lst.append(ds)
            elif ch == 'v':
                valid_lst.append(ds)
            elif ch == 't':
                test_lst.append(ds)
        return ConcatDataset(train_lst), ConcatDataset(valid_lst), ConcatDataset(test_lst)
    
    # 参数含义见 get_classification_datasets
    def split_by_event(self, label_lst, event_mask, include_trailing_seg): # mask含义同split_by_seg， include_trailing_seg表示最后一次发作之后的尾随间期
        assert(len(self) == len(label_lst))
        # TODO assert(r_right < v_left && v_right < t_left)
        seg_mask = []
        regex = r'[a-zA-Z]+([0-9+]+)-.+'
        for ds, label in zip(self, label_lst):
            match = re.search(regex, label)
            s = match.group(1)
            seg_mask.append(include_trailing_seg if s == '+' else event_mask[int(s)-1])
        return self.split_by_seg(self, seg_mask)
    

    def split_by_proportion(self, train_ratio=0.7, valid_ratio=0.1, shuffle=True, **kwargs): # 这个东西第三方库更好
        # 将一个ConcatDataset按比率分割，返回三个Subset
        num_samples = len(self)
        train_samples = int(train_ratio * num_samples)
        valid_samples = int(valid_ratio * num_samples)
        test_samples = num_samples - train_samples - valid_samples

        # train_sampler = SubsetRandomSampler(range(train_samples), generator=torch.Generator() if shuffle else None)
        # valid_sampler = SubsetRandomSampler(range(train_samples, train_samples + valid_samples), generator=torch.Generator() if shuffle else None)
        # test_sampler = SubsetRandomSampler(range(train_samples + valid_samples, num_samples), generator=torch.Generator() if shuffle else None)    
        # train_loader = DataLoader(self, sampler=train_sampler, **kwargs)
        # valid_loader = DataLoader(self, sampler=valid_sampler, **kwargs)
        # test_loader = DataLoader(self, sampler=test_sampler, **kwargs) 
        # return train_loader, valid_loader, test_loader

        # split_ratio = [train_ratio, valid_ratio, test_ratio] # TODO Error Handling
        # return random_split(self, split_ratio)

        return Subset(self, range(train_samples)), Subset(self, range(train_samples, train_samples + valid_samples)), \
                Subset(self, range(train_samples + valid_samples, num_samples))


class CLASSIFICATION_LABEL(IntEnum):
    INTER = -1,
    PRE = 0,
    ONSET = 1

class LabelDcrease(Dataset):
    def __init__(self, X, init_y, window_size, sample_rate):
        self.X = X; self.window_size = window_size
        L = X.shape[1] - window_size + 1
        self.y = init_y - (np.arange(L) / sample_rate)
    def __len__(self):
        return self.y.size
    def __getitem__(self, index):
        return self.X[:, index:(index+self.window_size)], self.y[index]

class LabelConstant(Dataset):
    def __init__(self, X, y, window_size):
        self.X = X; self.y = y; self.window_size = window_size
    def __len__(self):
        return self.X.shape[1] - self.window_size + 1
    def __getitem__(self, index):
        return self.X[:, index:(index+self.window_size)], self.y

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
            ds_lst.append(LabelConstant(X, label.PRE, window_size))            
        elif 'Inter' in seg["Label"]: 
            ds_lst.append(LabelConstant(X, label.INTER, window_size))
        elif 'Onset' in seg["Label"]:
            ds_lst.append(LabelConstant(X, label.ONSET, window_size))
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
            ds_lst.append(LabelDcrease(X, y0, window_size, sample_rate))
            label_lst.append(seg["Label"])
        elif 'Inter' in seg["Label"]: # 默认所有inter都超过限幅clamp，不然直接用get_simulation_dataloader更方便！
            npy_path = os.path.join(data_dir, seg["File"])
            X = np.load(npy_path).squeeze()
            X = X[:, slice(*seg["Span"])]
            y0 = seg["PreSec"]; assert(y0 > clamp or y0 == -1) # Debug
            ds_lst.append(LabelConstant(X, clamp, window_size))
            label_lst.append(seg["Label"])
        else: # 暂时不支持混合发作期进来 # TODO
            pass # raise ValueError from seg["Label"]
        
    return SplicedDataset(zip(ds_lst, label_lst))


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
        
        final_weights = [Tensor(size=len(dst), dtype=torch.double).fill_(self.weights[dst.y] / sz_counter[dst.y]) 
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

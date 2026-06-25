import torch
import numpy as np


a=np.ones([3,4])
a = torch.from_numpy(a)#转化为tensor
arr1 = a[:1]
arr2 = a[2:]
arr3 =  torch.cat((arr1, arr2), dim=0)
print(arr3)
print(arr3.shape)
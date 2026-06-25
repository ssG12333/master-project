import numpy as np
import torch
from typing import Tuple
import random


class SumTree:
    """SumTree data structure for efficient priority sampling"""

    def __init__(self, capacity):
        self.capacity = capacity                      # 树的容量，即叶子节点的数量，叶子节点存储的是单条经验的优先级，叶子节点的值是这条经验的优先级
        self.tree = np.zeros(2 * capacity - 1)        # 存储优先级累积和的数组
        self.data = np.zeros(capacity, dtype=object)  # 存储实际数据的数组，（例如经验回放中的状态、动作等）
        self.n_entries = 0                            # 当前存储的有效数据条数，当 n_entries 达到 capacity 时，新的数据会覆盖旧的数据，n_entries 重新从 0 开始计数
        self.pending_idx = set()                      # 记录待更新的索引集合

    def _propagate(self, idx, change):         # 当子叶子节点的优先级更新后，其父节点的优先级累积也会改变
        parent = (idx - 1) // 2                # idx当前节点的索引，change 优先级的变化量，parents：当前节点的父节点索引，可推倒左、右子节点的公式
        self.tree[parent] += change            # 更新父节点的优先级累积和
        if parent != 0:                        # 如果当前父节点不是根节点
            self._propagate(parent, change)    # 通过递归更新父节点的累积和，直到到达根节点

    def _retrieve(self, idx, s):                              # 递归查找优先级累积和为 s 的节点
        left = 2 * idx + 1                                    # left当前节点的左子节点的索引
        right = left + 1                                      # right当前节点的右子节点的索引

        if left >= len(self.tree):                            # 检查是否到达叶子节点
            return idx                                        # 如果到达叶子结点，即返回当前节点的索引

        if s <= self.tree[left]:                              # 如果没有达到叶子结点：如果s 小于或等于左子树的优先级累积和，说明目标节点在左子树中。
            return self._retrieve(left, s)                    # 调用 _retrieve(left, s)，继续在左子树中查找
        else:                                                 # 如果目标值 s 大于左子树的优先级累积和，说明目标节点在右子树中。
            return self._retrieve(right, s - self.tree[left]) # 调用 _retrieve(right, s - self.tree[left])，继续在右子树中查找

    def total(self):                               #返回树的总优先级累积和，即根节点的值
        return self.tree[0]

    def add(self, p, data):                        # 在SumTree中添加一条新数据，并为其分配一个初始优先级。p：新数据的初始优先级。data：要存储的实际数据
        idx = self.n_entries + self.capacity - 1   # 新数据在 self.tree 中的叶子节点索引
        self.pending_idx.add(self.n_entries)       # 将当前数据的索引添加到 pending_idx 集合中，表示这个节点的优先级需要更新，索引为 self.n_entries
        self.data[self.n_entries] = data           # 将数据存储在 self.data 数组中，索引为 self.n_entries
        self.update(idx, p)                        # 调用 update 方法，更新叶子节点的优先级

        self.n_entries += 1
        if self.n_entries >= self.capacity:        # n_entries 达到容量上限，重新从 0 开始，覆盖旧的数据
            self.n_entries = 0

    def update(self, idx, p):                      # 更新某个叶子节点的优先级，并递归更新其所有父节点的优先级累积和。idx：要更新的叶子节点的索引。p：新的优先级
        change = p - self.tree[idx]                # 新优先级与旧优先级的差值
        self.tree[idx] = p                         # 将叶子节点的优先级更新为新的值
        self._propagate(idx, change)               # 调用 _propagate 方法，递归更新所有父节点的优先级累积和

    def get(self, s):                                    # 根据目标优先级累积和s，找到对应的叶子节点，并返回其索引、优先级和数据
        idx = self._retrieve(0, s)                   # 调用 _retrieve 方法，从根节点开始查找，直到找到目标优先级累积和对应的叶子节点
        dataIdx = idx - self.capacity + 1                # 叶子节点在 self.data 数组中的索引
        self.pending_idx.discard(dataIdx)                # 如果 dataIdx 在 pending_idx 中，表示这个节点的优先级已经更新，将其从 pending_idx 中移除
        return (idx, self.tree[idx], self.data[dataIdx]) # 返回叶子节点的索引、优先级和对应的数据


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""

    def __init__(self, max_size: int = 1000000, state_dim: int = 15, action_dim: int = 4,  #alpha：优先级参数，控制优先级对采样概率的影响。alpha 越大，优先级对采样概率的影响越显著。
                 alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):    #beta： 重要性采样参数，用于纠正优先采样引入的偏差。beta 从 0 逐渐增加到 1；beta_increment:每次采样后 beta 的增量
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6                           #epsilon：一个小的常数，用于避免优先级为 0 的情况

        # SumTree for priority sampling 初始化SumTree,用于优先采样
        self.tree = SumTree(max_size)

        # Experience storage 初始化存储数组：用于存储状态、动作、奖励、下一个状态和是否完成标志
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

        self.ptr = 0   # 指针，指示当前写入位置，用于存储新数据。从 0 到 self.max_size - 1，循环使用。更新：每次添加新数据时增加 1，达到最大容量时重新从 0 开始。
        self.size = 0  # 记录当前缓冲区中存储的有效数据条数。范围：从 0 到 self.max_size，不会循环。更新：每次添加新数据时增加 1，达到最大容量时保持不变

    def add(self, state, action, reward, next_state, done, td_error=None):
        """Add experience to buffer with priority 添加一条新的经验到缓冲区，并计算其优先级"""
        # Store experience
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        # Calculate priority 计算优先级
        if td_error is None:  # 如果没有td_error，则将新经验的优先级设置为当前缓冲区中的最大优先级
            max_priority = np.max(self.tree.tree[-self.tree.capacity:]) if self.size > 0 else 1.0
            priority = max_priority     # 将新经验的优先级设置为 max_priority
            # 如果缓冲区中有数据（self.size > 0），则从 SumTree 的叶子节点中找到最大优先级；如果缓冲区为空，则默认最大优先级为 1.0
            # self.tree.tree[-self.tree.capacity:]即所有叶子节点的优先级
        else:
            priority = (np.abs(td_error) + self.epsilon) ** self.alpha  # 如果提供了td_error,则使用 TD 错误值计算优先级：priority=(abs(td_error)+ϵ)**α

        # Add to tree
        self.tree.add(priority, self.ptr) #将计算得到的优先级 priority 和当前指针位置 self.ptr 添加到 SumTree 中

        self.ptr = (self.ptr + 1) % self.max_size        #更新self.ptr和self.size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Tuple:
        """Sample batch with prioritized sampling"""
        if self.size == 0:   # 缓冲区不能为空
            raise ValueError("Cannot sample from empty buffer")

        batch_indices = []      # 用于存储采样数据的索引
        batch_priorities = []   # 用于存储采样数据的优先级

        segment = self.tree.total() / batch_size  # self.tree.total()：计算整个 SumTree 的总优先级，即根节点的值

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            try:
                (tree_idx, priority, data_idx) = self.tree.get(s)   # s：随机选择的优先级累积和
                if data_idx is not None and data_idx < self.size:   # 叶子节点索引data_idx在当前缓冲区的有效范围内
                    batch_indices.append(data_idx)                  # 将 data_idx 和对应的 priority 添加到 batch_indices 和 batch_priorities 中
                    batch_priorities.append(priority)
                else:                                                     # 对于无效的data_idx
                    fallback_idx = random.randint(0, self.size - 1)    # 随机选择一个有效的索引 fallback_idx 作为回退
                    batch_indices.append(fallback_idx)                    # 将 fallback_idx 添加到 batch_indices 中，并将优先级设置为 1.0（默认值）
                    batch_priorities.append(1.0)
            except:
                fallback_idx = random.randint(0, self.size - 1)     # 当try中出现异常时，随机选择一个有效的索引作为回退，并将优先级设置为 1.0
                batch_indices.append(fallback_idx)
                batch_priorities.append(1.0)

        # Calculate importance sampling weights 计算重要性采样权重
        batch_priorities = np.array(batch_priorities)
        sampling_probabilities = batch_priorities / (self.tree.total() + 1e-8)   ##计算采样概率

        weights = (self.size * sampling_probabilities + 1e-8) ** (-self.beta)    #计算重要性权重
        weights = weights / (np.max(weights) + 1e-8)                             # 将权重归一化，使得最大权重为 1

        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Get experiences
        batch_indices = np.array(batch_indices, dtype=int)

        return (
            self.state[batch_indices],
            self.action[batch_indices],
            self.reward[batch_indices],
            self.done[batch_indices],
            self.next_state[batch_indices],
            batch_indices,
            weights.reshape(-1, 1)          # weights.reshape(-1, 1)：将权重数组的形状调整为 (batch_size, 1)
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (np.abs(td_error) + self.epsilon) ** self.alpha
            tree_idx = idx + self.tree.capacity - 1
            self.tree.update(tree_idx, priority)

    def __len__(self):
        return self.size


class ReplayBuffer:
    '''Standard Experience replay buffer'''

    def __init__(self, max_size: int = 1000000, state_dim: int = 15, action_dim: int = 4):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):  #将一个新的经验（状态、动作、奖励、下一个状态、是否终止）添加到缓冲区中
        '''Add experience to buffer'''
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Tuple:  # 从缓冲区中随机采样一批数据
        '''Sample batch from buffer'''
        indices = np.random.randint(0, self.size, size=batch_size)  #随机生成 batch_size 个索引，范围从 0 到当前缓冲区的大小

        return (
            self.state[indices],
            self.action[indices],
            self.reward[indices],
            self.done[indices],
            self.next_state[indices]
        )    #使用这些索引从缓冲区中提取相应的数据元组

    def __len__(self):   #返回缓冲区中当前存储的数据量
        return self.size
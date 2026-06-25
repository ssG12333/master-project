from abc import ABC, abstractmethod # abc：Python 自带的“抽象基类”工具，用来定义接口；
import numpy as np
import torch

# 继承 ABC，说明这是一个抽象基类——不能被直接实例化，只能被其他具体算法（如 DDPG、TD3、SAC…）继承
# 功能：统一接口，让所有子类都必须实现同样一组方法（select_action、train、save、load）
class BaseAgent(ABC):
    '''Base class for RL agents'''

    def __init__(self, state_dim: int, action_dim: int, device: str = 'cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)

    @abstractmethod
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        '''Select action given state'''
        pass

    @abstractmethod
    def train(self, replay_buffer, batch_size: int) -> dict:
        '''Train the agent'''
        pass

    @abstractmethod
    def save(self, path: str):
        '''Save model'''
        pass

    @abstractmethod
    def load(self, path: str):
        '''Load model'''
        pass

#任何继承 BaseAgent 的具体算法都必须实现这四个方法（选动作、训练、保存、加载），否则运行时报错
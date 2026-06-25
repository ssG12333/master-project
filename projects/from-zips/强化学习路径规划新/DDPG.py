import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import time
import pygame
from matplotlib import pyplot as plt
from state import GameState

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

class CriticNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)
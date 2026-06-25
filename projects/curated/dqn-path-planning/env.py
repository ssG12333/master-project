import numpy as np
import warnings
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
thr = 10  # 阈值，表示到达终点的距离
v = 10  # 机器人每次移动距离
obstacle_width = 10  # 障碍物宽度
warnings.simplefilter("ignore", UserWarning)

final_route = {}

class Environment(object):
    def __init__(self, initial_position, target_position, X_max, Y_max, num_actions):
        self.state0 = np.zeros((2, 11, 11))
        self.state0[0][10][1] = 1  # 初始位置 (10, 1)

        self.Obstacle_x = [3,3, 3, 3, 3, 3,  3, 7, 7, 7, 7, 7,  7,7]
        self.Obstacle_y = [4,5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4, 5,6]

        self.vector_obstacle_x = [0] * len(self.Obstacle_x)
        self.vector_obstacle_y = [0] * len(self.Obstacle_x)

        for i in range(len(self.Obstacle_x)):
            self.vector_obstacle_x[i] = 10 * (self.Obstacle_x[i] - 0.5)
            self.vector_obstacle_y[i] = 10 * (10 - self.Obstacle_y[i] - 0.5)

        self.obstacle = [np.zeros((1, 4)).tolist() for i in range(len(self.Obstacle_x))]
        for i in range(len(self.vector_obstacle_x)):
            self.obstacle[i] = [self.vector_obstacle_x[i], self.vector_obstacle_y[i], obstacle_width, obstacle_width]

        for i in range(len(self.Obstacle_x)):
            self.state0[1, self.Obstacle_y[i], self.Obstacle_x[i]] = 1
        self.state0[1][0][9] = 1  # 终点位置

        self.X_max = X_max
        self.Y_max = Y_max
        self.vector_state0 = np.asarray(initial_position)
        self.Is_Terminal = False
        self.vector_agentState = np.copy(self.vector_state0)
        self.agentState = np.copy(self.state0)
        self.Terminal = np.asarray(target_position)
        self.doneType = 0
        self.max_episode_steps = 10000
        self.steps_counter = 0
        self.num_actions = num_actions

        self.dic = {}
        self.final_path = {}
        self.index = 0
        self.firstsuc = True
        self.longest = 0
        self.shortest = 0

        self.actionspace = {
            0: [v, 0], 1: [0, v], 2: [-v, 0], 3: [0, -v],
            4: [-v, v], 5: [-v, -v], 6: [v, v], 7: [v, -v]
        }

    def reset(self):
        self.agentState = np.copy(self.state0)
        self.vector_agentState = np.copy(self.vector_state0)
        self.dic = {}
        self.index = 0
        self.doneType = 0
        self.steps_counter = 0
        self.Is_Terminal = False

        # 更新障碍物和终点在状态数组中的标记
        self.state0 = np.zeros((2, 11, 11))
        self.state0[0][10][1] = 1  # 起点
        for i in range(len(self.Obstacle_x)):
            self.state0[1, self.Obstacle_y[i], self.Obstacle_x[i]] = 1
        terminal_x = int(self.Terminal[0] / 10)
        terminal_y = int(10 - self.Terminal[1] / 10)
        self.state0[1, terminal_y, terminal_x] = 1
        self.agentState = np.copy(self.state0)
        return self.agentState

    def step(self, action):
        V = self.actionspace[action]
        prev_state = np.copy(self.vector_agentState)
        self.vector_agentState[0] += V[0]
        self.vector_agentState[1] += V[1]

        # 检查边界
        if self.vector_agentState[0] < 0:
            self.vector_agentState[0] = 0
        if self.vector_agentState[0] > 100:
            self.vector_agentState[0] = 100
        if self.vector_agentState[1] < 0:
            self.vector_agentState[1] = 0
        if self.vector_agentState[1] > 100:
            self.vector_agentState[1] = 100

        # 检查障碍物碰撞
        if self.is_collision(self.vector_agentState):
            self.vector_agentState = prev_state
            reward = -20
            next_state_flag = 'obstacle'
            self.dic[self.index] = self.vector_agentState.tolist()
            self.index += 1
            i_x = np.copy(self.vector_agentState[0]) / 10
            i_y = 10 - np.copy(self.vector_agentState[1]) / 10
            self.agentState = np.copy(self.state0)
            self.agentState[0][9][1] = 0
            self.agentState[0, int(i_y), int(i_x)] = 1
            self.steps_counter += 1
            self.Is_Terminal = self.isTerminal()
            return self.agentState, next_state_flag, reward, self.Is_Terminal, None

        self.dic[self.index] = self.vector_agentState.tolist()
        self.index += 1
        i_x = np.copy(self.vector_agentState[0]) / 10
        i_y = 10 - np.copy(self.vector_agentState[1]) / 10
        self.agentState = np.copy(self.state0)
        self.agentState[0][9][1] = 0
        self.agentState[0, int(i_y), int(i_x)] = 1
        self.steps_counter += 1
        self.Is_Terminal = self.isTerminal()
        reward, next_state_flag = self.get_reward(self.vector_agentState, action)
        return self.agentState, next_state_flag, reward, self.Is_Terminal, None

    def isTerminal(self):
        Distance2Terminal = np.linalg.norm(np.subtract(self.vector_agentState, self.Terminal))
        if Distance2Terminal ** 0.5 == 0:
            self.doneType = 1
            return True
        return False

    def get_reward(self, state, action):
        reward = 0
        if not self.Is_Terminal:
            if self.is_collision(state):
                reward = -20
                next_state_flag = 'obstacle'
            else:
                if action in [0, 1, 2, 3]:
                    reward = -1
                else:
                    reward = -1.5
                next_state_flag = 'continue'
        elif self.doneType == 1:
            reward = 20
            next_state_flag = 'goal'
            if self.firstsuc:
                for j in range(len(self.dic)):
                    self.final_path[j] = self.dic[j]
                self.firstsuc = False
                self.longest = len(self.dic)
                self.shortest = len(self.dic)
            else:
                if len(self.dic) < len(self.final_path):
                    self.shortest = len(self.dic)
                    self.final_path = {}
                    for j in range(len(self.dic)):
                        self.final_path[j] = self.dic[j]
                if len(self.dic) > self.longest:
                    self.longest = len(self.dic)
        return reward, next_state_flag

    def final(self):
        print('最短路径:', self.shortest)
        print('最长路径:', self.longest)
        for j in range(len(self.final_path)):
            final_route[j] = self.final_path[j]

    def is_collision(self, state):
        delta = 0.5 * obstacle_width
        for (x, y, w, h) in self.obstacle:
            if 0 <= state[0] - (x - delta) <= w and 0 <= state[1] - (y - delta) <= h:
                return True
        return False

def final_states():
    return final_route
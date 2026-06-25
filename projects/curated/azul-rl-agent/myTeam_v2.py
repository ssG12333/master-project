from template import Agent
from Azul import azul_utils as utils
import random
from copy import deepcopy
import numpy as np
from collections import deque
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from Azul.azul_model import AzulGameRule as GameRule
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
THINKTIME   = 0.9
NUM_PLAYERS = 2

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        # DQN parameters
        self.state_size = 111  # Adjust this if necessary
        self.action_size = 150 # Adjust this if necessary
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.01
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        self.game_rule = GameRule(NUM_PLAYERS) # Agent stores an instance of GameRule, from which to obtain functions.
        self.round_number = 0  # Initialize round number

    def _build_model(self):
        # Build a simple neural network model
        model = Sequential()
        model.add(Dense(256,  activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_network(self):
        # Copy weights from the Q-Network to the target network
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        # Store experience in the replay buffer
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, actions):
        # Use epsilon-greedy strategy to select an action
        if np.random.rand() <= self.epsilon:
            return random.choice(actions)  # Randomly select an action

        act_values = self.model.predict(state)

        best_action_idx = np.argmax(act_values[0])

        if best_action_idx >= len(actions):
            print(f"Warning: best_action_idx {best_action_idx} is out of range. Selecting a random action.")
            return random.choice(actions)

        # Return the corresponding action from the actions list
        return actions[best_action_idx]

    def SelectAction(self, actions, game_state):
        state = self.preprocess_state(game_state)

        # Use Minimax to decide on action with some exploration
        if np.random.rand() <= self.epsilon:  # Exploration
            selected_action = random.choice(actions)
        else:  # Exploitation with Minimax
            selected_action = self.minimax_decision(state, actions, game_state, depth=3)

        # Execute the action, obtain the next state and reward
        next_state, reward, done = self.execute_action(selected_action, game_state)

        # Store the experience
        self.remember(state, actions.index(selected_action), reward, next_state, done)

        # Perform training less frequently
        if len(self.memory) > 64 and self.round_number % 5 == 0:  # Every 5 rounds
            self.replay(64)

        return selected_action

    def minimax_decision(self, state, actions, game_state, depth):
        """ Decide the best action using Minimax algorithm. """
        best_action = None
        best_value = -np.inf

        for action in actions:
            next_state, _, _ = self.execute_action(action, game_state)
            value = self.min_value(next_state, depth - 1, -np.inf, np.inf, game_state)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def max_value(self, state, depth, alpha, beta, game_state):
        if depth == 0 or self.game_rule.gameEnds():  # Terminal condition
            return self.evaluate_state(state)

        value = -np.inf
        actions = self.get_legal_actions(game_state, self.id)
        for action in actions:
            next_state, _, _ = self.execute_action(action, game_state)
            value = max(value, self.min_value(next_state, depth - 1, alpha, beta, game_state))
            if value >= beta:
                return value  # Beta cut-off
            alpha = max(alpha, value)
        return value

    def min_value(self, state, depth, alpha, beta, game_state):
        if depth == 0 or self.game_rule.gameEnds():  # Terminal condition
            return self.evaluate_state(state)

        value = np.inf
        opponent_id = 1 - self.id
        actions = self.get_legal_actions(game_state, opponent_id)
        for action in actions:
            next_state, _, _ = self.execute_action(action, game_state)
            value = min(value, self.max_value(next_state, depth - 1, alpha, beta, game_state))
            if value <= alpha:
                return value  # Alpha cut-off
            beta = min(beta, value)
        return value

    def evaluate_state(self, state):
        current_score = state[0]  # 当前玩家得分
        opponent_score = state[1]  # 对手得分
        completed_rows = sum(1 for row in state[2:7] if all(tile == 1 for tile in row))  # 计算已完成的行
        completed_cols = sum(1 for col in range(5) if all(state[2 + row][col] == 1 for row in range(5)))  # 计算已完成的列
        remaining_tiles_current = sum(state[7:12])  # 剩余瓷砖数量（当前玩家）
        remaining_tiles_opponent = sum(state[12:17])  # 剩余瓷砖数量（对手）
        floor_tiles = len(state[17])  # 地板线瓷砖数量

        # 计算得分差异
        score_difference = current_score - opponent_score

        # 启发式评估
        evaluation = (score_difference +
                      5 * completed_rows +  # 每完成一行加分
                      7 * completed_cols -  # 每完成一列加分
                      2 * remaining_tiles_opponent +  # 增加对手剩余瓷砖的惩罚
                      -3 * floor_tiles -  # 地板线瓷砖的惩罚
                      (remaining_tiles_current / 2)  # 当前剩余瓷砖的惩罚
                      )

        # 评估瓷砖放置得分
        evaluation += self.evaluate_tile_placement(state)

        return evaluation



    def evaluate_tile_placement(self, state):
        evaluation_score = 0

        for row in range(5):
            for col in range(5):
                if state[2 + row][col] == 1:  # 如果当前格子有瓷砖
                    horizontal_count = 1  # 包括自己
                    vertical_count = 1  # 包括自己

                    # 检查横向瓷砖
                    for c in range(col - 1, -1, -1):
                        if state[2 + row][c] == 1:
                            horizontal_count += 1
                        else:
                            break
                    for c in range(col + 1, 5):
                        if state[2 + row][c] == 1:
                            horizontal_count += 1
                        else:
                            break

                    # 检查纵向瓷砖
                    for r in range(row - 1, -1, -1):
                        if state[2 + r][col] == 1:
                            vertical_count += 1
                        else:
                            break
                    for r in range(row + 1, 5):
                        if state[2 + r][col] == 1:
                            vertical_count += 1
                        else:
                            break

                    # 计算得分
                    evaluation_score += horizontal_count + vertical_count

        return evaluation_score

    def get_legal_actions(self, game_state, player_id):
        """ Returns the list of legal actions for the player. """
        return self.game_rule.getLegalActions(game_state, player_id)

    def preprocess_state(self, game_state):
        round_number = self.round_number / 8  # Normalized round
        factory_state = [tile / 4.0 for factory in game_state.factories for tile in factory.tiles]
        centre_pool_state = [game_state.centre_pool.tiles[tile] / 20.0 for tile in utils.Tile]
        first_player_token = 1 if game_state.first_agent_taken else 0
        current_player_state = self.encode_player_state(game_state.agents[self.id])
        opponent_state = self.encode_player_state(game_state.agents[1 - self.id])
        bag_state = [game_state.bag.count(tile) / 100.0 for tile in utils.Tile]

        state_vector = [round_number] + factory_state + centre_pool_state + [first_player_token] \
                       + current_player_state + opponent_state + bag_state

        return np.reshape(state_vector, [1, self.state_size])

    def encode_player_state(self, player_state):
        encoded_state = []
        encoded_state.extend([player_state.grid_state[row][col] for row in range(5) for col in range(5)])
        encoded_state.extend([player_state.lines_number[line] / (line + 1) for line in range(5)])
        encoded_state.extend(player_state.floor + [0] * (7 - len(player_state.floor)))
        return encoded_state

    def execute_action(self, action, game_state):
        next_state = deepcopy(game_state)
        agent_state = next_state.agents[self.id]
        opponent_id = 1 - self.id
        opponent_state = next_state.agents[opponent_id]

        try:
            self.game_rule.generateSuccessor(next_state, action, self.id)
        except AssertionError:
            return self.preprocess_state(next_state), -10, True  # Penalty for illegal action

        self.round_number += 1
        reward = self.calculate_reward(agent_state, opponent_state, next_state)
        done = self.game_rule.gameEnds()

        return self.preprocess_state(next_state), reward, done

    def calculate_reward(self, agent_state, opponent_state, next_state):
        reward = 0

        # 得分变化奖励
        score_change = next_state.agents[self.id].score - agent_state.score
        reward += score_change  # 得分增加

        # 行列完成奖励
        for row in agent_state.grid_state:
            if all(tile == 1 for tile in row):  # 行完成
                reward += 10

        for col in range(5):
            if all(agent_state.grid_state[row][col] == 1 for row in range(5)):  # 列完成
                reward += 15

        # 瓷砖放置奖励
        placed_tiles = sum(sum(row) for row in next_state.agents[self.id].grid_state) - \
                       sum(sum(row) for row in agent_state.grid_state)
        opponent_tiles = sum(sum(row) for row in opponent_state.grid_state)

        # 鼓励多放砖块
        reward += placed_tiles * 2  # 每放置一个瓷砖增加奖励

        if placed_tiles > opponent_tiles:
            reward += 5  # 本轮放置的瓷砖数量超过对手

        # 地板线惩罚
        new_floor_tiles = len(next_state.agents[self.id].floor) - len(agent_state.floor)
        if new_floor_tiles > 0:
            reward -= 5 * new_floor_tiles  # 每个放置在地板线的瓷砖惩罚

        # 对手得分惩罚
        opponent_score_change = opponent_state.score - (agent_state.score - score_change)
        reward -= opponent_score_change * 0.5  # 对手得分变化惩罚

        # 新放置瓷砖的得分
        new_tile_score = self.calculate_tile_score(next_state, agent_state)
        reward += new_tile_score

        # 游戏结束奖励
        if self.game_rule.gameEnds():
            if next_state.agents[self.id].score > opponent_state.score:
                reward += 50  # 胜利奖励
            else:
                reward -= 50  # 失败惩罚

        return reward

    def calculate_tile_score(self, next_state, agent_state):
        score = 0
        new_tile_positions = []  # 存储新放置的瓷砖位置

        # 找到新放置的瓷砖
        for row in range(5):
            for col in range(5):
                if next_state.agents[self.id].grid_state[row][col] == 1 and \
                        agent_state.grid_state[row][col] == 0:  # 新放置的瓷砖
                    new_tile_positions.append((row, col))

        # 计算新放置的瓷砖得分
        for row, col in new_tile_positions:
            adjacent_tiles = 0
            horizontal_count = 0
            vertical_count = 0

            # 检查周围瓷砖
            if row > 0 and next_state.agents[self.id].grid_state[row - 1][col] == 1:  # 上方
                adjacent_tiles += 1
                vertical_count += 1
            if row < 4 and next_state.agents[self.id].grid_state[row + 1][col] == 1:  # 下方
                adjacent_tiles += 1
                vertical_count += 1
            if col > 0 and next_state.agents[self.id].grid_state[row][col - 1] == 1:  # 左侧
                adjacent_tiles += 1
                horizontal_count += 1
            if col < 4 and next_state.agents[self.id].grid_state[row][col + 1] == 1:  # 右侧
                adjacent_tiles += 1
                horizontal_count += 1

            # 根据规则计算得分
            if adjacent_tiles == 0:
                score += 1  # 无邻接瓷砖得1分
            else:
                score += (horizontal_count + 1)  # 包含新放置的瓷砖
                score += (vertical_count + 1)  # 包含新放置的瓷砖

        return score

    def replay(self, batch_size):
        # Train the model using a batch of experiences from the replay buffer
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            # Calculate the target Q-value
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            # Use the current Q-network to predict Q-values
            target_f = self.model.predict(state)
            # Update the Q-value for the chosen action
            target_f[0][action] = target
            # Train the Q-network to minimize the loss
            self.model.fit(state, target_f, epochs=1, verbose=0)
        # Decrease the exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

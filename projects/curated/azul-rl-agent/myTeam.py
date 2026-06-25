from template import Agent
from Azul import azul_utils as utils
import random
from copy import deepcopy
import numpy as np
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import os
from Azul.azul_model import AzulGameRule as GameRule
import tensorflow as tf

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
THINKTIME = 0.9
NUM_PLAYERS = 2


class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0

    def push(self, experience, priority):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.priorities.append(priority)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        return [self.buffer[idx] for idx in indices], indices

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        # DQN parameters
        self.state_size = 111  # Adjust this if necessary
        self.action_size = 150  # Adjust this if necessary
        self.memory = PrioritizedReplayBuffer(2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.01
        self.model_filename = os.path.join(os.path.dirname(__file__), 'DDQN_model.keras')

        try:
            self.model = self._load_or_create_model()
        except Exception as e:
            print(f"Error loading or creating model: {e}")
            print("Initializing a new model.")
            self.model = self._build_model()

        self.target_model = self._build_model()
        self.update_target_network()
        self.game_rule = GameRule(NUM_PLAYERS)
        self.round_number = 0  # Initialize round number

    def _load_or_create_model(self):
        if os.path.exists(self.model_filename):
            print(f"Loading existing model from {self.model_filename}")
            try:
                return tf.keras.models.load_model(self.model_filename)
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Creating new model instead.")
                return self._build_model()
        else:
            print(f"Model file not found at {self.model_filename}. Creating new model.")
            return self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        # Calculate a simple priority based on the reward
        priority = abs(reward) + 1e-5  # Add a small constant to avoid zero priority
        self.memory.push((state, action, reward, next_state, done), priority)

    def act(self, state, actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(actions)

        act_values = self.model.predict(state)
        best_action_idx = np.argmax(act_values[0])

        if best_action_idx >= len(actions):
            print(f"Warning: best_action_idx {best_action_idx} is out of range. Selecting a random action.")
            return random.choice(actions)

        return actions[best_action_idx]

    def SelectAction(self, actions, game_state):
        state = self.preprocess_state(game_state)

        if np.random.rand() <= self.epsilon:
            selected_action = random.choice(actions)
        else:
            selected_action = self.minimax_decision(state, actions, game_state, depth=3)

        next_state, reward, done = self.execute_action(selected_action, game_state)
        self.remember(state, actions.index(selected_action), reward, next_state, done)

        if len(self.memory.buffer) > 64 and self.round_number % 2 == 0:
            print("123345678")
            self.replay(64)

        if self.game_rule.gameEnds(game_state):
            self.save_model()

        return selected_action

    def minimax_decision(self, state, actions, game_state, depth):
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
        if depth == 0 or self.game_rule.gameEnds():
            return self.evaluate_state(state)

        value = -np.inf
        actions = self.get_legal_actions(game_state, self.id)
        for action in actions:
            next_state, _, _ = self.execute_action(action, game_state)
            value = max(value, self.min_value(next_state, depth - 1, alpha, beta, game_state))
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def min_value(self, state, depth, alpha, beta, game_state):
        if depth == 0 or self.game_rule.gameEnds():
            return self.evaluate_state(state)

        value = np.inf
        opponent_id = 1 - self.id
        actions = self.get_legal_actions(game_state, opponent_id)
        for action in actions:
            next_state, _, _ = self.execute_action(action, game_state)
            value = min(value, self.max_value(next_state, depth - 1, alpha, beta, game_state))
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    def evaluate_state(self, state):
        current_score = state[0]
        opponent_score = state[1]
        completed_rows = sum(1 for row in state[2:7] if all(tile == 1 for tile in row))
        completed_cols = sum(1 for col in range(5) if all(state[2 + row][col] == 1 for row in range(5)))
        remaining_tiles_current = sum(state[7:12])
        remaining_tiles_opponent = sum(state[12:17])
        floor_tiles = len(state[17])

        score_difference = current_score - opponent_score

        evaluation = (score_difference +
                      5 * completed_rows +
                      7 * completed_cols -
                      2 * remaining_tiles_opponent -
                      3 * floor_tiles -
                      (remaining_tiles_current / 2))

        evaluation += self.evaluate_tile_placement(state)

        return evaluation

    def evaluate_tile_placement(self, state):
        evaluation_score = 0

        for row in range(5):
            for col in range(5):
                if state[2 + row][col] == 1:
                    horizontal_count = 1
                    vertical_count = 1

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

                    evaluation_score += horizontal_count + vertical_count

        return evaluation_score

    def get_legal_actions(self, game_state, player_id):
        return self.game_rule.getLegalActions(game_state, player_id)

    def preprocess_state(self, game_state):
        round_number = self.round_number / 8
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
            return self.preprocess_state(next_state), -10, True

        self.round_number += 1
        reward = self.calculate_reward(agent_state, opponent_state, next_state)
        done = self.game_rule.gameEnds()

        return self.preprocess_state(next_state), reward, done

    def calculate_reward(self, agent_state, opponent_state, next_state):
        reward = 0

        current_score = next_state.agents[self.id].score
        opponent_score = opponent_state.score

        # 得分变化奖励
        score_change = current_score - agent_state.score
        if current_score > opponent_score:
            reward += score_change * 3  # 加大得分变化奖励

        # 行列完成奖励
        for row in agent_state.grid_state:
            if all(tile == 1 for tile in row):
                reward += 20  # 提高行完成奖励

        for col in range(5):
            if all(agent_state.grid_state[row][col] == 1 for row in range(5)):
                reward += 25  # 提高列完成奖励

        # 瓷砖放置奖励
        placed_tiles = sum(sum(row) for row in next_state.agents[self.id].grid_state) - \
                       sum(sum(row) for row in agent_state.grid_state)
        opponent_tiles = sum(sum(row) for row in opponent_state.grid_state)

        reward += placed_tiles * 3  # 增加放置瓷砖的奖励

        if placed_tiles > opponent_tiles:
            reward += 10  # 增加超过对手的奖励

        # 地板线惩罚
        new_floor_tiles = len(next_state.agents[self.id].floor) - len(agent_state.floor)
        if new_floor_tiles > 0:
            reward -= 10 * new_floor_tiles  # 更大的惩罚

        # 游戏结束奖励
        if self.game_rule.gameEnds():
            if current_score > opponent_score:
                reward += 100  # 胜利奖励
            else:
                reward -= 100  # 失败惩罚

        return reward

    def calculate_tile_score(self, next_state, agent_state):
        score = 0
        new_tile_positions = []

        for row in range(5):
            for col in range(5):
                if next_state.agents[self.id].grid_state[row][col] == 1 and \
                        agent_state.grid_state[row][col] == 0:
                    new_tile_positions.append((row, col))

        for row, col in new_tile_positions:
            adjacent_tiles = 0
            horizontal_count = 0
            vertical_count = 0

            if row > 0 and next_state.agents[self.id].grid_state[row - 1][col] == 1:
                adjacent_tiles += 1
                vertical_count += 1
            if row < 4 and next_state.agents[self.id].grid_state[row + 1][col] == 1:
                adjacent_tiles += 1
                vertical_count += 1
            if col > 0 and next_state.agents[self.id].grid_state[row][col - 1] == 1:
                adjacent_tiles += 1
                horizontal_count += 1
            if col < 4 and next_state.agents[self.id].grid_state[row][col + 1] == 1:
                adjacent_tiles += 1
                horizontal_count += 1

            if adjacent_tiles == 0:
                score += 1
            else:
                score += (horizontal_count + 1)
                score += (vertical_count + 1)

        return score

    def replay(self, batch_size):
        if len(self.memory.buffer) < batch_size:
            return

        minibatch, indices = self.memory.sample(batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])

            target_f = self.model.predict(state)
            target_f[0][action] = target

            # Update the priority based on the TD error
            priority = abs(target - target_f[0][action]) + 1e-5
            self.memory.update_priorities(indices, [priority for _ in indices])

            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self):
        try:
            print(f"Saving model to {self.model_filename}")
            self.model.save(self.model_filename)
            print("Model saved successfully.")
        except Exception as e:
            print(f"Error saving model: {e}")

    def __del__(self):
        print("Agent is being destroyed.")
        if hasattr(self, 'model'):
            print("Saving the model...")
            self.save_model()
        else:
            print("No model to save.")



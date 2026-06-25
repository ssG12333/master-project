import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Tuple, List

from envs.evtol_env import EVTOLEnvironment
from envs.obstacles import TrainingStage
from agents.td3 import TD3Agent
from utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class CurriculumTrainer:
    '''Curriculum learning trainer for eVTOL with detailed analysis'''

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize environment with first stage
        self.current_stage = TrainingStage.BASIC_FLIGHT
        self.env = EVTOLEnvironment(self.current_stage,config)

        agent_params = config['agent_params']

        # 1. 强制转换为 float，确保 PyTorch 接受
        lr_actor = float(agent_params['lr_actor'])
        lr_critic = float(agent_params['lr_critic'])

        # 2. 从字典中移除这两个键，避免它们作为字符串重复传递
        del agent_params['lr_actor']
        del agent_params['lr_critic']

        # 3. Initialize agent
        self.agent = TD3Agent(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim,
            max_action=1.0,
            device=self.device,
            # 将转换后的 float 值显式传递给 Agent
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            **agent_params  # 将字典中剩余的参数安全解包
                 # config 是一个字典，通常用于存储配置参数（超参数）；config['agent_params']是字典中的一个键值对，其中 agent_params 是一个子字典，专门用于存储与智能体（Agent）相关的参数
        )                                 # ** 是一个解包操作符，用于将字典中的键值对解包为关键字参数

        # Initialize replay buffer with PER option
        use_per = config.get('use_prioritized_replay', True)   #config.get用于安全地从字典中获取键值对的值。
        if use_per:                                            # 如果键 'use_prioritized_replay' 存在于字典中，返回对应的值。如果键 'use_prioritized_replay' 不存在于字典中，返回默认值 True。
            self.replay_buffer = PrioritizedReplayBuffer(
                max_size=config['buffer_size'],
                state_dim=self.env.state_dim,
                action_dim=self.env.action_dim,
                alpha=config.get('per_alpha', 0.6),
                beta=config.get('per_beta', 0.4),
                beta_increment=config.get('per_beta_increment', 0.001)
            )
            print(
                f"Using Prioritized Experience Replay (α={config.get('per_alpha', 0.6)}, β={config.get('per_beta', 0.4)})")
        else:
            self.replay_buffer = ReplayBuffer(
                max_size=config['buffer_size'],
                state_dim=self.env.state_dim,
                action_dim=self.env.action_dim
            )
            print("Using Standard Experience Replay")

        # Logging 初始化summaryWriter，指定存储路径
        self.writer = SummaryWriter(log_dir=os.path.join(config['output_dir'], 'tensorboard'))

        # Training/validation alternation 每10000次训练就验证100次
        self.train_steps_per_cycle = config.get('train_steps_per_cycle', 10000)
        self.validation_episodes_per_cycle = config.get('validation_episodes_per_cycle', 100)

        # Stage progression criteria 训练阶段成功阈值
        stage_progression = config.get('stage_progression', {})
        self.stage_success_threshold = {
            TrainingStage.BASIC_FLIGHT: stage_progression.get('BASIC_FLIGHT', 0.9),
            TrainingStage.SIMPLE_NAV: stage_progression.get('SIMPLE_NAV', 0.9),
            TrainingStage.COMPLEX_NAV: stage_progression.get('COMPLEX_NAV', 0.7),
            TrainingStage.GENERALIZATION: stage_progression.get('GENERALIZATION', None)
        }

        # 计数器

        self.best_reward = -float('inf')   #初始化迄今为止的最大奖励值，设置为负无穷，确保任何负值奖励值都比它大
        self.episode_count = 0              # 回合计数器
        self.validation_episode_count = 0   # 验证回合计数器
        self.training_step_count = 0        #  训练步数计数器

        # 阶段自适应最大步数配置
        self.max_episode_steps_per_stage = config.get('max_episode_steps_per_stage', {
            'BASIC_FLIGHT': 800,
            'SIMPLE_NAV': 1000,
            'COMPLEX_NAV': 1500,
            'GENERALIZATION': 2500
        })
        self.default_max_steps = config.get('max_episode_steps', 1500)

    def train(self):
        '''Main training loop with interleaved validation 验证训练交错循环逻辑'''
        total_env_steps = 0

        while total_env_steps < self.config['max_steps']:
            # Training phase
            print(
                f"\n--- Training Phase: Steps {total_env_steps} to {total_env_steps + self.train_steps_per_cycle} ---")

            cycle_env_steps = 0    # 当前训练周期内的步数计数器

            while cycle_env_steps < self.train_steps_per_cycle and total_env_steps < self.config['max_steps']:

                episode_reward, episode_steps = self.run_training_episode()
                cycle_env_steps += episode_steps   # 将当前episode的步数加到当前周期的总步数中
                total_env_steps += episode_steps   # 将当前episode的步数加到总训练步数中
                self.episode_count += 1

                # 中频日志：每个Episode的基本信息
                self.log_episode_metrics(self.episode_count, episode_reward, episode_steps)

                # 每个episode只训练一次，而不是每一步都训练
                if len(self.replay_buffer) > self.config['warmup_steps']:
                    # 每个episode训练min(episode_steps, 10)次，避免过度训练
                    train_steps = min(episode_steps, 10)
                    for _ in range(train_steps):
                        training_metrics = self.agent.train(self.replay_buffer, self.config['batch_size'])
                        self.training_step_count += 1

                        # 高频日志：每次训练更新的指标
                        self.log_training_step_metrics(training_metrics, self.training_step_count)


            # Validation phase
            print(f"\n--- Validation Phase: {self.validation_episodes_per_cycle} episodes ---")
            validation_metrics = self.run_validation_phase()    # 运行验证阶段，在无探索的情况下测试当前策略的性能，并返回验证指标

            # 低频详细验证分析
            self.output_detailed_validation_analysis(validation_metrics)

            # Check stage progression
            success_threshold = self.stage_success_threshold.get(self.current_stage)
            if success_threshold and validation_metrics['success_rate'] >= success_threshold:   #  success_threshold为True
                self.progress_to_next_stage()

            # Save checkpoint
            if self.episode_count % self.config.get('save_freq', 100) == 0:   #每100个episode保存一次
                self.save_checkpoint(self.episode_count)

        self.writer.close()

    def run_training_episode(self) -> Tuple[float, int]:
        '''Run single training episode'''
        state = self.env.reset()
        episode_reward = 0.0
        episode_steps = 0

        max_steps_for_stage = self.max_episode_steps_per_stage.get(
            self.current_stage.name,
            self.default_max_steps
        )

        while episode_steps < max_steps_for_stage:
            if len(self.replay_buffer) < self.config['warmup_steps']:
                action = np.random.uniform(-1, 1, self.env.action_dim)
            else:
                action = self.agent.select_action(state, evaluate=False)   # 返回带噪声的动作（训练阶段）

            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store transition with TD error calculation for PER
            if hasattr(self.replay_buffer, 'update_priorities'):                    # hasattr(object, name)，object: 要检查的对象。name: 要检查的属性或方法名称（字符串形式）。如果对象有该属性或方法，返回 True；如果没有，返回 False
                if len(self.replay_buffer) < self.config['warmup_steps']:           # 预热阶段
                    td_error = 1.0
                else:
                    td_error = self.agent.calculate_td_error(state, action, reward, next_state, done)
                self.replay_buffer.add(state, action, reward, next_state, done, td_error)  #将transition样本和TD误差一起存入优先经验回放缓冲区
            else:
                self.replay_buffer.add(state, action, reward, next_state, done)  #如果缓冲区不支持优先经验回放，只存储基本的转移样本，不带优先级信息

            episode_reward += reward
            episode_steps += 1
            state = next_state

            if done:
                break     #如果结束，提前跳出循环，终止当前episode

        return episode_reward, episode_steps

    def run_validation_phase(self) -> Dict:   # 运行验证阶段的方法
        '''Run validation phase and return comprehensive aggregated metrics 运行验证阶段并返回综合的聚合指标'''
        all_episodes_data = []   # 初始化一个空列表，用于存储所有验证episode的数据

        for i in range(self.validation_episodes_per_cycle):
            self.validation_episode_count += 1
            episode_data = self.run_validation_episode()  #运行单个验证episode，并返回该episode的详细数据
            all_episodes_data.append(episode_data)        # 将当前episode的数据添加到总列表中

        # Aggregate validation metrics with Q-value analysis
        aggregated = self.aggregate_validation_metrics(all_episodes_data) # 调用聚合方法，将所有验证episode的数据聚合成一个综合的指标字典，计算平均性能和各种统计量

        # Save best model
        if aggregated['avg_reward'] > self.best_reward:   # 检查当前验证的平均奖励是否超过了历史最佳奖励
            self.best_reward = aggregated['avg_reward']   # 更新最佳奖励值为当前的平均奖励
            self.save_checkpoint(0, is_best=True)  # 保存当前模型为最佳模型

        return aggregated

    def run_validation_episode(self) -> Dict:     # 运行单个验证episode的方法
        '''Run single validation episode with comprehensive analysis including Q-values'''
        state = self.env.reset()
        episode_reward = 0.0
        episode_steps = 0
        trajectory = []
        q_value_data = []
        action_data = []
        state_data = []

        max_steps_for_stage = self.max_episode_steps_per_stage.get(
            self.current_stage.name,
            self.default_max_steps
        )

        while episode_steps < max_steps_for_stage:
            # Get action with Q-value information for analysis
            if hasattr(self.agent, 'select_action_with_analysis'):   # 检查智能体是否支持分析模式
                # 如果智能体支持分析模式
                action_info = self.agent.select_action_with_analysis(state, evaluate=True)
                action = action_info['action']  # 从返回信息中提取动作（无噪声动作）
                q_values = action_info.get('q_values', None)
                action_confidence = action_info.get('confidence', None)
            else:
                # 标准模式：只获取动作，然后单独计算Q值
                action = self.agent.select_action(state, evaluate=True)

                # 手动获取Q值用于分析
                if len(self.replay_buffer) > self.config['warmup_steps']:   # 过了预热阶段
                    q_values = self.agent.get_q_values(state, action)       # 手动计算当前状态的动作和q值以及动作置信度
                    action_confidence = self.calculate_action_confidence(state)
                else:   # 如果还在预热阶段
                    q_values = None
                    action_confidence = 0.0

            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store data for analysis
            trajectory.append(info['position'].copy())
            state_data.append(state.copy())
            action_data.append(action.copy())

            if q_values is not None:  # 如果有q值信息
                q_value_data.append({
                    'q1': q_values.get('q1', 0.0),
                    'q2': q_values.get('q2', 0.0),
                    'q_min': q_values.get('q_min', 0.0),
                    'confidence': action_confidence
                })

            episode_reward += reward
            episode_steps += 1
            state = next_state

            if done:
                break

        # Calculate comprehensive episode metrics 计算综合的episode指标
        episode_data = self.calculate_comprehensive_episode_metrics(
            trajectory, state_data, action_data, q_value_data,
            episode_reward, episode_steps, info
        )

        # 添加障碍物统计信息
        if hasattr(self.env, 'current_obstacle_stats'):
            episode_data.update({
                'obstacle_count': self.env.current_obstacle_stats['count'],   # 障碍物数量信息
                'obstacle_density': self.env.current_obstacle_stats['density_actual'],  # 障碍物密度信息
                'avg_obstacle_height': self.env.current_obstacle_stats['avg_height'],   # 障碍物平均高度信息
            })

        return episode_data

    def calculate_action_confidence(self, state: np.ndarray) -> float:
        '''手动计算动作选择的置信度（基于Q值的一致性）'''
        try:
            # 验证阶段，随机生成10个动作样本
            sample_actions = np.random.uniform(-1, 1, (10, self.env.action_dim))
            q_values = []

            for action in sample_actions:
                q_vals = self.agent.get_q_values(state, action)
                if q_vals:
                    q_values.append(q_vals.get('q_min', 0.0))  # 将Q值的最小值（q_min）添加到列表中，如果没有则使用默认值0.0

            if len(q_values) > 1:
                # 使用Q值的标准差的倒数作为置信度指标
                q_std = np.std(q_values) # 计算所有Q值的标准差；标准差越小，说明不同动作的Q值越接近，决策越不确定
                confidence = 1.0 / (1.0 + q_std)
            else:  #如果没有获取到足够的Q值，使用默认的中间置信度
                confidence = 0.5

            return float(confidence)
        except:
            return 0.5

    def calculate_comprehensive_episode_metrics(
            self, trajectory: List, state_data: List, action_data: List,
            q_value_data: List, reward: float, steps: int, info: Dict
    ) -> Dict:     # 计算单个episode综合指标的方法
        '''Calculate comprehensive episode metrics including Q-value analysis'''
        trajectory = np.array(trajectory)

        # Basic metrics
        success = info['distance_to_goal'] < self.env.target_radius  # 成功标准：距离小于目标半径
        d_safe = self.config.get('environment', {}).get('collision_distance_threshold', 7.5)
        collision = not success and (                                   # 碰撞标准：没成功且高度<0或与障碍物距离小于翼展一半
                self.env.dynamics.position[2] < 0 or
                self.env.obstacle_gen.get_nearest_obstacle(self.env.dynamics.position)[0] < d_safe
        )


        # Path metrics
        if len(trajectory) > 1:  #避免轨迹只有一个点
            path_length = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))  # axis=0：沿着行的方向操作（计算相邻位置的差值）；axis=1：沿着列的方向操作（计算每个差向量的欧几里得长度）
            path_efficiency = np.linalg.norm(trajectory[-1] - trajectory[0]) / path_length if path_length > 0 else 0.0   # 直线距离与实际飞行距离的比值，衡量飞行路径的优化程度
        else:                                   # trajectory[-1]：目标点位置；trajectory[0]：终止点位置
            path_length = 0.0
            path_efficiency = 0.0

        # Safety analysis 安全覆盖率
        safety_distances = []
        for pos in trajectory: #对于航迹中的每一个点
            min_dist, _ = self.env.obstacle_gen.get_nearest_obstacle(pos)   # min_dist：到最近障碍物的距离;_：障碍物信息（这里用 _ 表示不关心这个值）
            safety_distances.append(min_dist)   # safety_distances：存储每个位置的最小安全距离

        # 计算安全距离阈值，确保20m
        config_safety_threshold = self.config.get('environment', {}).get('safety_distance_threshold', 20.0)
        safety_threshold = config_safety_threshold
        safety_coverage = np.mean(np.array(safety_distances) > safety_threshold) if safety_distances else 0.0  #safety_coverage：飞行安全覆盖率。当到障碍物的最小距离大于一倍翼展时（约15m），才安全。
        min_safety_distance = min(safety_distances) if safety_distances else float('inf')  # 最小安全距离：智能体在整个飞行过程中最接近障碍物的那次距离

        # Q-value analysis
        q_metrics = {}
        if q_value_data:
            q1_values = [q['q1'] for q in q_value_data]   # 提取所有q1值
            q2_values = [q['q2'] for q in q_value_data]   # 提取所有q2值
            q_min_values = [q['q_min'] for q in q_value_data]  # 提取所有q值
            confidence_values = [q['confidence'] for q in q_value_data]   # 提取所有confidence_values值，两个Critic的Q值差异衡量指标

            q_metrics = {
                'avg_q1': np.mean(q1_values),  # 求平均值
                'avg_q2': np.mean(q2_values),
                'avg_q_min': np.mean(q_min_values),
                'q_value_stability': 1.0 - np.std(q_min_values) / (abs(np.mean(q_min_values)) + 1e-8),  # 衡量q值的稳定性，如果波动很大，可能需要调整学习率。1 - (标准差 / 绝对值平均值)。接近1.0表示很稳定，接近0.0波动很大。+ 1e-8：防止除零错误的小常数
                'avg_action_confidence': np.mean(confidence_values),    #动作置信度指标：反映智能体的决策质量
                'q_value_trend': self.calculate_q_trend(q_min_values),   #q值在整个spisode中的变化趋势，反映任务难度和学习进度。Q值在上升，说明智能体越做越好；Q值在下降，说明遇到困难。
            }

        # Action analysis  动作评估指标
        action_metrics = {}
        if action_data:
            action_array = np.array(action_data)
            action_metrics = {
                'action_magnitude': np.mean(np.linalg.norm(action_array, axis=1)),    # 动作幅度（如果幅度不合适，可能需要调整动作范围）：计算每个动作向量的欧几里得范数，然后取平均值。值很大：动作很"激烈"，控制指令变化大值很小：动作很"温和"，控制指令变化小。需要根据具体任务调整，既不能太大也不能太小
                'action_smoothness': self.calculate_action_smoothness(action_array),  # 动作平滑度：影响飞行舒适性和安全性。如果平滑度太低：可能需要增加动作变化惩罚
                'action_diversity': self.calculate_action_diversity(action_array),    # 动作多样性：确保充分探索控制空间。如果多样性太低：可能需要增加探索噪声
            }

        # Combine all metrics
        episode_metrics = {
            # Basic metrics
            'success': success,
            'collision': collision,
            'reward': reward,
            'steps': steps,

            # Path metrics
            'path_length': path_length,
            'path_efficiency': path_efficiency,

            # Safety metrics
            'safety_coverage': safety_coverage,
            'min_safety_distance': min_safety_distance,
        }

        # Add Q-value and action metrics
        episode_metrics.update(q_metrics)
        episode_metrics.update(action_metrics)

        return episode_metrics

    def calculate_q_trend(self, q_values: List[float]) -> float:
        '''计算Q值趋势（上升/下降）'''
        if len(q_values) < 2:
            return 0.0

        # 简单线性回归斜率
        x = np.arange(len(q_values))
        slope = np.corrcoef(x, q_values)[0, 1] if len(set(q_values)) > 1 else 0.0   # set(q_values)：去除重复值，检查Q值是否有变化如果所有Q值都相同（方差为0），相关系数无法计算，返回0.0
        return float(slope)  # 正趋势：智能体越做越好，逐渐掌握任务。负趋势：智能体遇到困难，性能下降。无趋势：学习停滞，可能需要调整策略

    def calculate_action_smoothness(self, actions: np.ndarray) -> float:
        '''计算动作平滑度'''
        if len(actions) < 2:
            return 1.0

        action_changes = np.diff(actions, axis=0)  # 动作变化量
        smoothness = 1.0 / (1.0 + np.mean(np.linalg.norm(action_changes, axis=1)))   # smoothness = 1.0 / (1.0 + 平均变化幅度)
        return float(smoothness)

    def calculate_action_diversity(self, actions: np.ndarray) -> float:
        '''计算动作多样性'''
        if len(actions) < 2:
            return 0.0

        # 使用动作向量的标准差作为多样性指标
        diversity = np.mean(np.std(actions, axis=0))   # 沿着时间维度计算标准差
        return float(diversity)

    def aggregate_validation_metrics(self, episodes_data: List[Dict]) -> Dict:
        '''Aggregate comprehensive validation episode metrics  将多个验证episode的数据聚合成总体统计指标'''
        if not episodes_data:  # episodes_data不能为空
            return {}

        # Helper function to safely get and average metrics
        def safe_average(key):  # 定义一个内部辅助函数safe_average，用于安全计算平均值
            values = [ep[key] for ep in episodes_data if key in ep and ep[key] is not None]
            return np.mean(values) if values else 0.0

        def safe_std(key): # 定义另一个内部辅助函数safe_std，用于安全计算标准差 ，values = [ep[key] for ep in episodes_data if key in ep and ep[key] is not None]
            values = [ep[key] for ep in episodes_data if key in ep and ep[key] is not None]
            return np.std(values) if values else 0.0

        # Basic aggregated metrics 基础聚合指标
        aggregated = {  # 初始化聚合指标
            # Performance metrics 性能指标
            'success_rate': np.mean([ep['success'] for ep in episodes_data]), # 成功率
            'collision_rate': np.mean([ep['collision'] for ep in episodes_data]),  # 碰撞率
            'avg_reward': safe_average('reward'),     # 平均奖励
            'std_reward': safe_std('reward'),         # 奖励标准差
            'avg_steps': safe_average('steps'),       # 平均步数

            # Path metrics
            'avg_path_length': safe_average('path_length'),
            'avg_path_efficiency': safe_average('path_efficiency'),

            # Safety metrics
            'avg_safety_coverage': safe_average('safety_coverage'),
            'min_safety_distance_overall': min([ep.get('min_safety_distance', float('inf'))
                                                for ep in episodes_data
                                                if ep.get('min_safety_distance', float('inf')) != float('inf')],
                                               default=float('inf')),
        }

        # Q-value metrics (if available)
        q_value_episodes = [ep for ep in episodes_data if 'avg_q1' in ep]
        if q_value_episodes:
            aggregated.update({
                'avg_q1_overall': np.mean([ep['avg_q1'] for ep in q_value_episodes]),
                'avg_q2_overall': np.mean([ep['avg_q2'] for ep in q_value_episodes]),
                'avg_q_min_overall': np.mean([ep['avg_q_min'] for ep in q_value_episodes]),
                'avg_q_stability': np.mean([ep['q_value_stability'] for ep in q_value_episodes]),
                'avg_action_confidence': np.mean([ep['avg_action_confidence'] for ep in q_value_episodes]),
                'avg_q_trend': np.mean([ep['q_value_trend'] for ep in q_value_episodes]),
            })

        # Action metrics (if available)
        action_episodes = [ep for ep in episodes_data if 'action_magnitude' in ep]
        if action_episodes:
            aggregated.update({
                'avg_action_magnitude': np.mean([ep['action_magnitude'] for ep in action_episodes]),
                'avg_action_smoothness': np.mean([ep['action_smoothness'] for ep in action_episodes]),
                'avg_action_diversity': np.mean([ep['action_diversity'] for ep in action_episodes]),
            })

        # Obstacle environment metrics
        obstacle_episodes = [ep for ep in episodes_data if 'obstacle_count' in ep]
        if obstacle_episodes:
            aggregated.update({
                'avg_obstacle_count': np.mean([ep['obstacle_count'] for ep in obstacle_episodes]),
                'avg_obstacle_density': np.mean([ep['obstacle_density'] for ep in obstacle_episodes]),
                'avg_obstacle_height': np.mean([ep['avg_obstacle_height'] for ep in obstacle_episodes]),
            })

        return aggregated

    def log_training_step_metrics(self, metrics: Dict, step: int):   # 定义一个高频日志，记录训练步骤指标的方法：metrics: Dict：包含各种训练指标的字典。step: int：当前训练步数
        '''高频日志：每次训练步骤的指标'''
        for key, value in metrics.items():
            self.writer.add_scalar(f'Training_Step/{key}', value, step)   # Tenserboard可视化

    def log_episode_metrics(self, episode_num: int, reward: float, steps: int):
        '''中频日志：每个Episode的基本指标'''
        print(f"Episode {episode_num} | Stage: {self.current_stage.name} | "
              f"Reward: {reward:.1f} | Steps: {steps}")

        self.writer.add_scalar('Episode/reward', reward, episode_num)
        self.writer.add_scalar('Episode/steps', steps, episode_num)

    def output_detailed_validation_analysis(self, metrics: Dict):
        '''详细验证分析输出，包含Q值分析'''
        print(f"\n" + "=" * 80)  #重复=80次
        print(f"COMPREHENSIVE VALIDATION ANALYSIS - Episode {self.episode_count}")
        print(f"Stage: {self.current_stage.name}")
        print("=" * 80)

        # Performance metrics
        print(" PERFORMANCE METRICS")
        print(f"  Success Rate:        {metrics['success_rate']:.2%}")
        print(f"  Collision Rate:      {metrics['collision_rate']:.2%}")
        print(f"  Average Reward:      {metrics['avg_reward']:.1f} ± {metrics['std_reward']:.1f}")
        print(f"  Average Steps:       {metrics['avg_steps']:.1f}")

        # Path metrics
        print("\n  PATH METRICS")
        print(f"  Average Path Length: {metrics['avg_path_length']:.1f}m")
        print(f"  Path Efficiency:     {metrics['avg_path_efficiency']:.2%}")

        # Safety metrics
        print("\n  SAFETY METRICS")
        print(f"  Safety Coverage:     {metrics['avg_safety_coverage']:.2%}")
        min_dist = metrics['min_safety_distance_overall']
        if min_dist != float('inf'):
            print(f"  Min Safety Distance: {min_dist:.1f}m")
        else:
            print(f"  Min Safety Distance: No close approaches")

        # Q-value analysis (if available)   #q值分析
        if 'avg_q1_overall' in metrics:
            print("\n Q-VALUE ANALYSIS")
            print(f"  Average Q1:          {metrics['avg_q1_overall']:.2f}")
            print(f"  Average Q2:          {metrics['avg_q2_overall']:.2f}")
            print(f"  Average Q-min:       {metrics['avg_q_min_overall']:.2f}")
            print(f"  Q-Value Stability:   {metrics['avg_q_stability']:.2%}")
            print(f"  Action Confidence:   {metrics['avg_action_confidence']:.2%}")
            trend = metrics['avg_q_trend']
            trend_desc = "↗ Increasing" if trend > 0.1 else "↘ Decreasing" if trend < -0.1 else "➡ Stable"
            print(f"  Q-Value Trend:       {trend_desc} ({trend:.3f})")

        # Action analysis (if available)
        if 'avg_action_magnitude' in metrics:
            print("\n⚡ ACTION ANALYSIS")
            print(f"  Action Magnitude:    {metrics['avg_action_magnitude']:.3f}")
            print(f"  Action Smoothness:   {metrics['avg_action_smoothness']:.2%}")
            print(f"  Action Diversity:    {metrics['avg_action_diversity']:.3f}")

        # Environment metrics
        if 'avg_obstacle_count' in metrics:
            print("\n  ENVIRONMENT METRICS")
            print(
                f"  Obstacles:           {metrics['avg_obstacle_count']:.1f} (density: {metrics['avg_obstacle_density']:.2e}/m²)")
            print(f"  Avg Obstacle Height: {metrics['avg_obstacle_height']:.1f}m")

        print("=" * 80)

        # Log all metrics to tensorboard
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value) and value != float('inf'):
                self.writer.add_scalar(f'Validation/{key}', value, self.episode_count)

    def progress_to_next_stage(self):
        '''Progress to next training stage'''
        stage_map = {
            TrainingStage.BASIC_FLIGHT: TrainingStage.SIMPLE_NAV,
            TrainingStage.SIMPLE_NAV: TrainingStage.COMPLEX_NAV,
            TrainingStage.COMPLEX_NAV: TrainingStage.GENERALIZATION
        }

        if self.current_stage in stage_map:
            self.current_stage = stage_map[self.current_stage]
            print(f"\n STAGE PROGRESSION: Moving to {self.current_stage.name} \n")
            self.env = EVTOLEnvironment(self.current_stage, self.config)

    def save_checkpoint(self, episode_num: int, is_best: bool = False):
        '''Save checkpoint'''
        checkpoint_dir = os.path.join(self.config['output_dir'], 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)     # 创建检查点目录，如果已存在则不报错

        if is_best:
            path = os.path.join(checkpoint_dir, 'best_model.pt')   # 如果是最佳模型，用固定的最佳模型文件
        else:
            path = os.path.join(checkpoint_dir, f'checkpoint_ep{episode_num}.pt')    # 如果不是最佳模型，使用包含回合数的文件名

        self.agent.save(path)      # 调用agent的save方法保存模型到指定路径
        print(f" Checkpoint saved: {path}")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from parameters import *


class TaskEnv:
    def __init__(self,seed = None):
        self.agent_num = AGENT_NUM
        self.task_num = TASK_NUM
        self.max_requirements = MAX_REQUIREMENTS
        self.execution_time_range = EXECUTION_TIME_RANGE

        if seed is not None:
            np.random.seed(seed)

        self.task_dic, self.agent_dic, self.depot = self.generate_env()

        self.max_waiting_time = 10  # 智能体在任务点等待其他智能体来一起执行任务的最大等待时间

        self.current_time = 0
        self.dt = 0.2               # 画图时对应self.current_time的时间步长
        self.finished = False

    def generate_env(self):
        depot = np.random.rand(1,2)[0,:]
        task_dic = dict()
        agent_dic = dict()
        for i in range(self.task_num):
            require = np.random.randint(1, self.max_requirements + 1)
            task_dic[i] = {
                'ID': i,
                'requirements': require,                                            # 任务所需的智能体数量，不变量
                'state': require,                                                   # 还需要的智能体数量，总需求量减去已经选择了该任务的智能体数量
                'members': [],                                                      # 选择了该任务的智能体，存储ID
                'location': np.random.rand(1,2)[0,:],
                'feasible_assignment': False,                                       # 这项任务分配是否可行，是否可以开始任务执行
                'finished': False,                                                  # 任务是否完成
                'time_start': np.inf,
                'time_finish': np.inf,
                # 任务执行完所需的时间
                'required_time': np.random.randint(self.execution_time_range[0], self.execution_time_range[1] + 1),
            }
        for i in range(self.agent_num):
            agent_dic[i] = {
                'ID': i,
                'location': depot,  #选择目标以后会瞬移到目标点
                'route': [],
                'arrival_time': [], #到达每个目标的到达时间
                'velocity': 0.2,
                'travel_time': 0,
                'next_decision': 0, # 下一次决策时间
                'travel_dist': 0,
                'working_condition': 0,
                'assigned': False,  # 是否正在执行任务，感觉后面可以删掉
                'returned': False,  # 是否已经返回仓库了
            }
        depot = {'location': depot,
                 'members': [],
                 'ID': -1}
        return task_dic, agent_dic, depot

    @staticmethod
    def get_matrix(dictionary, key): # 获取字典中指定键的值,返回列表
        key_matrix = []
        for value in dictionary.values():
            key_matrix.append(value[key])
        return key_matrix

    # 找到'next_decision'最小的智能体作为下一个要做决定的智能体列表，并返回其需要做决定的时间，
    # 如果'next_decision'为空，则说明所有智能体都以指向仓库了，返回空智能体列表和最晚'arrival_time'的时间即可，整个流程也结束了
    def next_decision(self):
        decision_time = np.array(self.get_matrix(self.agent_dic, 'next_decision'))
        if np.all(np.isnan(decision_time)):
            return [],max(map(lambda x: max(x) if x else 0, self.get_matrix(self.agent_dic, 'arrival_time')))
        next_decision = np.nanmin(decision_time)
        agents = np.where(decision_time == next_decision)[0]
        return agents, next_decision

    # 把agents按坐标是否相同进行划分,传入
    # 输入：agents = [0, 1, 2, 3]
    # 输出：[[0, 1, 3], [2]]
    def get_unique_group(self, agents):
        location = np.array(self.get_matrix(self.agent_dic, 'location'))[agents]
        unique_location = np.unique(location, axis=0)   # 去除重复的坐标
        unique_group = []
        for loc in unique_location:
            unique_group.append(agents[np.where(np.all(location == loc, axis=1))[0].tolist()].tolist())
        return unique_group

    def get_arrival_time(self, agent_id, task_id):          # 找到智能体最近一次到达task_id的时间
        arrival_time = self.agent_dic[agent_id]['arrival_time']
        arrival_for_task = np.where(np.array(self.agent_dic[agent_id]['route']) == task_id)[0][-1]
        return float(arrival_time[arrival_for_task])

    def task_update(self):
        for task in self.task_dic.values():
            if not task['feasible_assignment']:
                abilities = len(task['members'])
                # 得到到达了该任务的所有智能体的到达时间
                arrival = np.array([self.get_arrival_time(member, task['ID']) for member in task['members']])
                task['status'] = task['requirements'] - abilities
                # 任务点的智能体数量足够的话
                if task['status'] <= 0:
                    # 如果该任务的智能体达到时间都在最大等待时间内的话，就认为该任务点上有足够的任务点，任务分配可行，可以开始执行任务
                    # 否则超出时间的智能体就。。。
                    if np.max(arrival) - np.min(arrival) <= self.max_waiting_time:
                        task['time_start'] = float(np.max(arrival))
                        task['time_finish'] = float(np.max(arrival) + task['required_time'])
                        task['feasible_assignment'] = True
                    else:
                        # 如果选择任务的数量足够的话，把等不到最晚到达的智能体的智能体都剔除
                        infeasible_members = arrival <= np.max(arrival) - self.max_waiting_time
                        for member in np.array(task['members'])[infeasible_members]:
                            task['members'].remove(member)
                else:
                    # 剔除等待时间超时的智能体
                    for member in task['members']:
                        if self.current_time - self.get_arrival_time(member, task['ID']) >= self.max_waiting_time:
                            task['members'].remove(member)
            else:
                if self.current_time >= task['time_finish']:
                    task['finished'] = True

        for member in self.depot['members']:
            if self.current_time >= self.get_arrival_time(member, -1):
                self.agent_dic[member]['returned'] = True

    def agent_update(self):
        for agent in self.agent_dic.values():
            if len(agent['arrival_time']) > 0:
                if agent['route'][-1] == -1:
                    agent['next_decision'] = np.nan
                else:
                    current_task = self.task_dic[agent['route'][-1]]
                    if current_task['feasible_assignment'] and not current_task['finished']:
                        agent['next_decision'] = float(current_task['time_finish'])
                        if self.current_time >= float(current_task['time_start']):
                            agent['assigned'] = True
                    else:
                        agent['next_decision'] = self.get_arrival_time(agent['ID'], current_task['ID']) + self.max_waiting_time
                        agent['assigned'] = False

    # 返回已经完成的任务，1为完成，0为未完成
    def get_finished_task_mask(self):
        unfinished_tasks = []
        for task in self.task_dic.values():
            unfinished_tasks.append(task['status'] > 0)
        mask = np.logical_not(unfinished_tasks)
        return mask

    @staticmethod
    def calculate_eulidean_distance(agent, task):           # 计算两个点之间的欧式距离(标量)
        return np.linalg.norm(agent['location'] - task['location'])

    def agent_step(self, agent_id, target):
        agent = self.agent_dic[agent_id]
        if target != -1:
            task = self.task_dic[target]
        else:
            task = self.depot
        agent['route'].append(target)
        agent['travel_time'] = self.calculate_eulidean_distance(agent, task) / agent['velocity']
        agent['arrival_time']+= [self.current_time + agent['travel_time']]
        agent['location'] = task['location']
        if agent_id not in task['members']:
            task['members'].append(agent_id)


    def step(self, group, leader_id, target):
        vacancy = self.task_dic[target]['status'] if target in self.task_dic.keys() else len(group)
        group.remove(leader_id)
        available_agents = len(group)
        if vacancy > 1:
            followers = np.random.choice(group, np.minimum(vacancy-1, available_agents), False).tolist()
            for follower in followers:
                group.remove(follower)
            members = [leader_id] + followers
        else:
            members = [leader_id]
        for member in members:
            self.agent_step(member, target)
        return group

    def check_finished(self):
        return np.all(self.get_matrix(self.agent_dic, 'returned'))

    # 返回所有智能体的3种时间、a['assigned']和与参数中的agent的xy距离差
    def get_current_agent_status(self, agent):
        status = []
        for a in self.agent_dic.values():
            if len(a['route']) > 0 and a['route'][-1] in self.task_dic.keys():
                # 智能体到达任务还需多少时间，最小为0
                travel_time = np.clip(self.get_arrival_time(a['ID'], a['route'][-1]) - self.current_time, a_min=0, a_max=None)
                # 智能体到达任务等待的时间，最小为0
                current_waiting_time = np.clip(self.current_time - self.get_arrival_time(a['ID'], a['route'][-1]),a_min=0, a_max=None) if self.current_time <= self.task_dic[a['route'][-1]]['time_start'] else 0
                # 智能体开始任务后，完成任务还需多少时间，最小为0
                remaining_working_time = np.clip(self.task_dic[a['route'][-1]]['time_finish'] - self.current_time, a_min=0, a_max=None) if self.current_time >= self.task_dic[a['route'][-1]]['time_start'] else 0
            else:
                travel_time = 0
                current_waiting_time = 0
                remaining_working_time = 0
            temp_status = np.hstack([travel_time, current_waiting_time, remaining_working_time, agent['location'] - a['location'], a['assigned']])
            status.append(temp_status)
        current_agents = np.vstack(status)
        return current_agents

    # 返回所有任务的还需的智能体数量、所需的智能体数量、任务持续时间和与参数中的agent的xy距离差，包含仓库
    def get_current_task_status(self, agent):
        status = []
        for t in self.task_dic.values():
            temp_status = np.hstack([t['status'], t['requirements'], t['required_time'], t['location'] - agent['location']])
            status.append(temp_status)
        status = [np.hstack([0, 0, 0, self.depot['location'] - agent['location']])] + status
        current_tasks = np.vstack(status)
        return current_tasks

    def plot_figure(self, save_path='task_env.gif', fps=5):
        def node_location(node_id):
            if node_id == -1:
                return self.depot['location']
            return self.task_dic[node_id]['location']

        def build_timelines():
            """
            为每个智能体重建轨迹段：
            每一段包含 起点、终点、出发时刻、到达时刻。
            overlap_key 用于判定“重叠智能体”的颜色：
            前一个目标相同、当前目标相同、到达时间相同。
            """
            timelines = {}
            overlap_members = {}

            for agent_id, agent in self.agent_dic.items():
                segments = []
                prev_target = -1
                prev_loc = self.depot['location'].copy()

                for idx, target in enumerate(agent['route']):
                    dst_loc = node_location(target).copy()
                    arrival_time = float(agent['arrival_time'][idx])

                    dist = np.linalg.norm(dst_loc - prev_loc)
                    travel_time = dist / agent['velocity'] if agent['velocity'] > 0 else 0.0
                    start_time = float(arrival_time - travel_time)

                    overlap_key = (prev_target, target, round(arrival_time, 8))

                    segments.append({
                        'src': prev_target,
                        'dst': target,
                        'src_loc': prev_loc.copy(),
                        'dst_loc': dst_loc.copy(),
                        'start_time': start_time,
                        'end_time': arrival_time,
                        'overlap_key': overlap_key,
                    })

                    if overlap_key not in overlap_members:
                        overlap_members[overlap_key] = set()
                    overlap_members[overlap_key].add(agent_id)

                    prev_target = target
                    prev_loc = dst_loc.copy()

                timelines[agent_id] = segments

            overlap_size = {k: len(v) for k, v in overlap_members.items()}
            return timelines, overlap_size

        def agent_state_at(segments, t, total_time):
            """
            返回某个时刻智能体的状态：
            - moving: 正在迁移
            - static: 已到达任务点并停留
            - None: 到达仓库后隐藏
            """
            if len(segments) == 0:
                return None

            eps = 1e-9

            for i, seg in enumerate(segments):
                start_t = seg['start_time']
                end_t = seg['end_time']
                next_start_t = segments[i + 1]['start_time'] if i + 1 < len(segments) else total_time + self.dt

                # 正在迁移
                if start_t - eps <= t < end_t - eps and end_t > start_t + eps:
                    alpha = (t - start_t) / (end_t - start_t)
                    pos = seg['src_loc'] + alpha * (seg['dst_loc'] - seg['src_loc'])
                    return {
                        'position': pos,
                        'moving': True,
                        'start_loc': seg['src_loc'],
                        'dst': seg['dst'],
                        'overlap_key': seg['overlap_key'],
                    }

                # 已到达当前目标并停留
                if (end_t - eps <= t < next_start_t - eps) or (i == len(segments) - 1 and t >= end_t - eps):
                    if seg['dst'] == -1:
                        # 到达仓库后完全遮掉
                        return None
                    return {
                        'position': seg['dst_loc'],
                        'moving': False,
                        'start_loc': seg['src_loc'],
                        'dst': seg['dst'],
                        'overlap_key': seg['overlap_key'],
                    }

            return None

        timelines, overlap_size = build_timelines()

        latest_arrival = 0.0
        for agent in self.agent_dic.values():
            if len(agent['arrival_time']) > 0:
                latest_arrival = max(latest_arrival, max(agent['arrival_time']))

        latest_finish = 0.0
        for task in self.task_dic.values():
            latest_finish = max(latest_finish, float(task['time_finish']))

        total_time = max(latest_arrival, latest_finish, float(self.current_time))
        frames = np.arange(0.0, total_time + self.dt, self.dt)
        if len(frames) == 0:
            frames = np.array([0.0])

        # 不同重叠数量对应不同颜色，4个及以上共用一种
        overlap_color = {
            1: '#FF0000',  # 纯红
            2: '#3CB44B',  # 翠绿
            3: '#00CED1',  # 深天蓝
            4: '#9400D3',  # 亮紫色
        }

        fig, ax = plt.subplots(figsize=(7, 7))

        def update(frame_idx):
            t = float(frames[frame_idx])
            ax.cla()

            ax.set_xlim(-0.08, 1.08)
            ax.set_ylim(-0.08, 1.08)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Time = {t:.1f}')

            # 1) 先画未完成任务点（橙色，层级低于智能体）
            for task in self.task_dic.values():
                is_finished = task['feasible_assignment'] and (t >= float(task['time_finish']))
                if not is_finished:
                    x, y = task['location']
                    ax.scatter(
                        x, y,
                        s=280,
                        c='orange',
                        marker='o',
                        edgecolors='black',
                        linewidths=1.2,
                        zorder=3
                    )
                    ax.text(
                        x, y,
                        str(task['requirements']),
                        ha='center',
                        va='center',
                        fontsize=10,
                        color='black',
                        zorder=7
                    )

            # 2) 再画智能体与迁移连线（在未完成任务点上方，在完成任务点/仓库下方）
            for agent_id, segments in timelines.items():
                state = agent_state_at(segments, t, total_time)
                if state is None:
                    continue

                count = min(overlap_size.get(state['overlap_key'], 1), 4)
                color = overlap_color[count]

                x, y = state['position']

                # 迁移时从起点到当前位置画线
                if state['moving']:
                    sx, sy = state['start_loc']
                    ax.plot(
                        [sx, x], [sy, y],
                        color=color,
                        linewidth=1.6,
                        alpha=0.9,
                        zorder=2
                    )

                ax.scatter(
                    x, y,
                    s=170,
                    c=color,
                    marker='^',
                    edgecolors='black',
                    linewidths=0.8,
                    zorder=4
                )

            # 3) 再画已完成任务点（绿色，盖住智能体）
            for task in self.task_dic.values():
                is_finished = task['feasible_assignment'] and (t >= float(task['time_finish']))
                if is_finished:
                    x, y = task['location']
                    ax.scatter(
                        x, y,
                        s=280,
                        c='green',
                        marker='o',
                        edgecolors='black',
                        linewidths=1.2,
                        zorder=5
                    )
                    ax.text(
                        x, y,
                        str(task['requirements']),
                        ha='center',
                        va='center',
                        fontsize=10,
                        color='black',
                        zorder=7
                    )

            # 4) 最后画仓库（蓝色方块，盖住到达仓库的智能体）
            dx, dy = self.depot['location']
            ax.scatter(
                dx, dy,
                s=340,
                c='royalblue',
                marker='s',
                edgecolors='black',
                linewidths=1.2,
                zorder=6
            )
            ax.text(
                dx, dy,
                'D',
                ha='center',
                va='center',
                fontsize=10,
                color='white',
                fontweight='bold',
                zorder=7
            )

            return []

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(frames),
            interval=max(1, int(1000 / fps)),
            blit=False,
            repeat=False
        )

        writer = animation.PillowWriter(fps=fps)
        ani.save(save_path, writer=writer)
        plt.close(fig)

        print(f'GIF saved to: {save_path}')
        return ani
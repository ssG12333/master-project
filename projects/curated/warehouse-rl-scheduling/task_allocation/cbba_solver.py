import numpy as np
import copy


class CBBASolver:
    def __init__(self, agent_dic, task_dic, depot, velocity=0.2, max_waiting_time=10):
        self.agent_dic = copy.deepcopy(agent_dic)
        self.task_dic = copy.deepcopy(task_dic)
        self.depot = copy.deepcopy(depot)
        self.velocity = velocity
        self.max_waiting_time = max_waiting_time

        self.num_agents = len(self.agent_dic)
        self.num_tasks = len(self.task_dic)

        # CBBA 内部状态
        self.bundle = {i: [] for i in range(self.num_agents)}        # 每个智能体的任务束
        self.path = {i: [] for i in range(self.num_agents)}          # 每个智能体的路径
        self.y = np.zeros(self.num_tasks)                            # 全局最高出价
        self.z = np.full(self.num_tasks, -1, dtype=int)              # 最高出价对应的智能体

    @staticmethod
    def calculate_distance(loc1, loc2):
        return np.linalg.norm(np.array(loc1) - np.array(loc2))

    def calculate_score(self, agent_id, task_id, current_path):
        agent = self.agent_dic[agent_id]
        task = self.task_dic[task_id]

        # 计算加入该任务后的总路径长度
        if len(current_path) == 0:
            dist = self.calculate_distance(agent['location'], task['location'])
        else:
            last_task_id = current_path[-1]
            if last_task_id == -1:
                last_loc = self.depot['location']
            else:
                last_loc = self.task_dic[last_task_id]['location']
            dist = self.calculate_distance(last_loc, task['location'])

        travel_time = dist / self.velocity
        score = 1.0 / (travel_time + 1e-8)  # 分数为时间的倒数

        # 考虑任务需求的满足程度
        requirements = task['requirements']
        if requirements > 0:
            score *= requirements  # 需求越大的任务优先级越高

        return score

    def solve(self, max_iterations=50):
        for iteration in range(max_iterations):
            changed = False

            for agent_id in range(self.num_agents):
                agent = self.agent_dic[agent_id]
                current_bundle = self.bundle[agent_id]
                current_path = self.path[agent_id]

                # Phase 1: Bundle Construction
                best_task = -1
                best_score = 0
                best_insert_idx = -1

                for task_id in range(self.num_tasks):
                    if task_id in current_bundle:
                        continue

                    task = self.task_dic[task_id]
                    if task['finished'] or task['requirements'] <= 0:
                        continue

                    # 尝试将任务插入到路径的不同位置
                    for insert_idx in range(len(current_path) + 1):
                        test_path = current_path[:insert_idx] + [task_id] + current_path[insert_idx:]

                        score = self.calculate_score(agent_id, task_id, test_path[:insert_idx + 1])

                        if score > best_score:
                            best_score = score
                            best_task = task_id
                            best_insert_idx = insert_idx

                if best_task != -1:
                    self.bundle[agent_id].append(best_task)
                    self.path[agent_id].insert(best_insert_idx, best_task)
                    self.y[best_task] = best_score
                    self.z[best_task] = agent_id
                    changed = True

                # Phase 2: Conflict Resolution (简化版 - 全连接网络)
                for task_id in range(self.num_tasks):
                    if self.z[task_id] == -1:
                        continue

                    winner = self.z[task_id]
                    winner_score = self.y[task_id]

                    # 检查是否有其他智能体出价更高
                    for other_id in range(self.num_agents):
                        if other_id == winner:
                            continue
                        if task_id in self.bundle[other_id]:
                            other_score = self.calculate_score(other_id, task_id, self.path[other_id])
                            if other_score > winner_score:
                                self.y[task_id] = other_score
                                self.z[task_id] = other_id
                                changed = True

            if not changed:
                break

        return self.path

    def execute_assignment(self, env):
        path_list = self.solve()

        # 按照路径顺序执行任务
        current_time = 0
        env.current_time = current_time

        # 为每个智能体设置路径
        for agent_id, path in path_list.items():
            agent = env.agent_dic[agent_id]
            for target in path:
                env.current_time = current_time
                env.agent_step(agent_id, target)
                current_time += agent['travel_time']
                env.task_update()
                env.agent_update()

        # 最后返回仓库
        for agent_id in range(env.agent_num):
            env.agent_step(agent_id, -1)

        return path_list, env.current_time

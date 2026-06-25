import json
import networkx as nx
from geopy.distance import geodesic
from typing import List, Dict
import time
import csv
import pandas as pd
import os
import ast
import requests
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder


class WuhanMultimodalDataBuilder:
    def __init__(self, api_key: str, jscode: str, gamma):
        self.api_key = api_key
        self.jscode = jscode
        self.graph = nx.Graph()
        self._transport_modes = { #_仅供内部使用的约定
            "metro": {"type_code": "150500"},
            "bus": {"type_code": "150700"},
        }
        self._wait_car_time = 8  # min

        self._bus_speed = 20 / 3.6  # m/s
        self._metro_speed = 80 / 3.6  # m/s
        self._walk_speed = 1.2  # m/s

        self._bus_price = 0.16  # 元/km
        self._metro_price = 0.2  # 元/km

        self.bound_location1 = (114.14302851571348, 30.514240287208427)
        self.bound_location2 = (114.36410022933235,30.400153757775985)

        self.i = 1
        self.goal = None
        self.current = None
        self.count = 0  # 一局游戏的步数
        self.reach_goal = 500
        self.gamma = gamma

        if os.path.exists("wuhan_multimodal_network.gexf"):
            # 优先读取文件夹下的gexf格式，然后建立网络
            self.graph = nx.read_gexf("wuhan_multimodal_network.gexf")
            self.filter_region_edge(self.bound_location1, self.bound_location2)
        else:
            # 文件夹下没网络的话，读取csv文件建立网络
            self.build_network()
            # self.visualize_network()
            # 将网络存储
            self.save_network()

        self.encoder = LabelEncoder()

    def _call_amap_api(self, url: str, params: Dict) -> Dict:
        """统一API调用方法（含基础参数）"""
        time.sleep(1)
        print(self.i)
        self.i = self.i + 1
        base_params = {"key": self.api_key, "city": "420100", "output": "JSON"}
        params.update(base_params)
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return json.loads(response.text)
        except Exception as e:
            print(f"API调用失败: {str(e)}")
            return {}

    def get_public_transport_data(self, city_adcode="420100", keywords="", mode=""):
        # 公交线路查询（支持分页获取所有线路）
        # 调用api返回 公交和地铁的路线数据
        url = "https://restapi.amap.com/v3/bus/linename?s=rsv3"
        params = {
            "key": self.api_key,
            "jscode": self.jscode,
            "city": city_adcode,
            "keywords": keywords,
            "output": "json",
            "offset": 1,
            "extensions": "all",
            "platform": "JS",
        }
        data = self._call_amap_api(url, params)  # 返回公交路线的数据
        result = []
        # 只取第一个
        if data.get("buslines") is None or len(data["buslines"]) == 0:
            print(data)
            return result
        for line in data["buslines"]:
            if (
                    mode == "bus"
                    and line["type"] == "地铁"
                    or mode == "metro"
                    and line["type"] != "地铁"
            ):
                continue
            line = data["buslines"][0]
            transport_data = {
                "Name": line.get("name"),
                "Name": line.get("name"),
                "Bus id": line.get("id"),
                "City id": city_adcode,
                "Company": line.get("company", "未知公司"),
                "Start station": line.get("start_stop"),
                "Stop station": line.get("end_stop"),
                "Distance": float(line.get("distance", 0)),  # 总长
                "Line": [],
                "Passing stations": [],
            }

            # 解析线路坐标
            if polyline := line.get("polyline"):
                transport_data["Line"] = [
                    list(map(float, coord.split(","))) for coord in polyline.split(";")
                ]

                # 解析途径站点
            for station in line.get("busstops", []):
                transport_data["Passing stations"].append(
                    {
                        "id": station.get("id"),
                        "location": station.get("location"),
                        "name": station.get("name"),
                        "sequence": station.get("sequence"),
                    }
                )

            result.append(transport_data)
            break
        return result

    def get_public_transport_stops(self, mode: str) -> List[Dict]:
        """获取公共交通站点（地铁/公交）"""
        # 获取地铁和公交信息
        # 通过读取当前目录下txt文档，
        # 存储线路到s数组中然后调取get_public_transport_data函数得到线路站点坐标
        stops = []
        s = ""
        if mode == "bus":
            with open("wuhan_bus.txt", "r+", encoding="utf-8") as f:
                s = [
                    ["公交" + part for part in i.strip().split(",")]
                    for i in f.readlines()
                ]
        elif mode == "metro":
            with open("wuhan_metro.txt", "r+", encoding="utf-8") as f:
                s = [i[:-1].split(",") for i in f.readlines()]
        for line in s[0]:
            result = self.get_public_transport_data(keywords=line, mode=mode)
            if len(result) != 0:
                stops.append(result[0])
        return stops

    def get_dist(self, stop1, stop2):
        """
        计算两个经纬度之间的距离
        :param stop1: str类型的经纬度对"22.324,11.23"
        :param stop2:
        :return: 距离m
        """
        if isinstance(stop1[0], str):
            stop1 = list(reversed(eval(stop1)))
        if isinstance(stop2[0], str):
            stop2 = list(reversed(eval(stop2)))
        # 计算地理距离（米）
        dist = geodesic(
            stop1, # stop可以为list[纬度，精度]，也可以是tuple(纬度，精度)
            stop2,
        ).meters
        return dist

    def _add_transfer_edges(self, max_walk_distance=500):
        """自动添加换乘连接边（论文中的超网络构建）
            公交和地铁的换乘，默认可以两公里内到的，都能换乘 没有添加公交与gongjiao
        """
        all_nodes = list(self.graph.nodes(data=True))
        for i in range(len(all_nodes)):
            node1 = all_nodes[i]
            node1_id = node1[0]
            for j in range(i + 1, len(all_nodes)):
                node2 = all_nodes[j]
                node2_id = node2[0]
                # 不同车站换乘
                if node1[1]["type"] == node2[1]["type"]:
                    continue
                # if node1[1]["line"] == node2[1]["line"]:
                #     continue
                dist = self.get_dist(node1[1]["location"], node2[1]["location"])
                # 通过距离限制站内换乘
                if not self.graph.has_edge(node1_id, node2_id) and dist <= max_walk_distance:
                    transfer_time = self.calculate_transfer_time(node1_id, node2_id)
                    self.graph.add_edge(
                        node1[0],
                        node2[0],
                        mode="walk",
                        time=transfer_time,
                        cost=0,
                        line="metro_bus_connection",
                    )

    def build_trans_network(self, mode: str):
        '''
            建立networkx网络
            1. 数据获取：如果有csv文件，则直接读取csv文件，如果没有则通过api湖片区数据
            2. 网络建立
        '''

        path = "metro.csv" if mode == "metro" else "bus.csv"
        lines = []
        if os.path.exists(path):
            # csv文件读取
            with open(path, mode="r", encoding="utf-8") as file:
                # 创建DictReader对象
                reader = csv.DictReader(file)
                # 遍历每一行
                for row in reader:
                    # 将每一行的字典添加到数据列表中
                    lines.append(row)
        else:
            # csv不存在的话，使用高德api获取对应的数据，并存储为csv文件
            lines = self.get_public_transport_stops(mode)
            df = pd.DataFrame(lines)
            df.to_csv(path, index=False, encoding="utf-8")
        for line in lines:
            # 筛选公交车信息
            # if mode == "bus" and np.random.random() > 0.01:
            #     continue

            # 同一路车的站点信息
            stations = (
                line["Passing stations"]
                if type(line["Passing stations"]) == list
                else ast.literal_eval(line["Passing stations"]) # 用作将字符串转变为可能的数据类型
            )
            for station in stations:
                if self.graph.has_node(station["id"]):
                    # 同站换乘点（该站已经通过其他线路建立过了节点） 通过虚拟节点表示
                    count = self.graph.nodes[station["id"]]["count"]
                    self.graph.nodes[station["id"]]["count"] += 1
                    station_id = station["id"] + str(count)
                    self.graph.add_node(
                        station_id,
                        **{
                            "type": mode,
                            "name": station["name"],
                            "location": station["location"],
                            "sequence": station["sequence"],
                            "line": line["Name"],
                        },
                    )
                    for i in range(0, self.graph.nodes[station["id"]]["count"]):
                        for j in range(i + 1, self.graph.nodes[station["id"]]["count"]):
                            # 同站换乘点的边建立
                            # 换乘点是通过虚拟点建立的，所以需要遍历所有的虚拟点，建立换乘的边
                            first_id = (
                                station["id"] + str(i) if i != 0 else station["id"]
                            )
                            second_id = (
                                station["id"] + str(j) if j != 0 else station["id"]
                            )

                            if not self.graph.has_edge(first_id, second_id):
                                dist = self.get_dist(
                                    self.graph.nodes[first_id]["location"],
                                    self.graph.nodes[second_id]["location"],
                                )

                                transfer_time = self.calculate_transfer_time(first_id, second_id)

                                self.graph.add_edge(
                                    first_id,
                                    second_id,
                                    mode="walk", # walk：步行，肯定是换乘了 metro：地铁 bus：公交
                                    time=transfer_time,  # 步行速度1.2m/s -> 分钟
                                    cost=0,
                                    line="same_" + mode + "_connection", # 就是这个边的类型,包括same_metro_connection same_bus_connection metro_bus_connection 线路名称表示同线路直达
                                )
                    station["id"] = station_id
                else:
                    self.graph.add_node(
                        station["id"],
                        **{
                            "type": mode,
                            "name": station["name"],
                            "location": station["location"],
                            "sequence": station["sequence"],
                            "line": line["Name"],
                            "count": 1,
                        },
                    )
            # 同线路的边的建立
            self._add_station_line(line["Name"], stations, mode)

    def build_network(self):
        """构建论文中的多式交通超网络"""
        for mode in self._transport_modes:
            self.build_trans_network(mode)
        # 添加换乘连接 最多支持0.5公里内的步行，要不然数据太多
        self._add_transfer_edges(500)
        self.filter_region_edge(self.bound_location1, self.bound_location2)

    def _add_station_line(self, line_name: str, stations_order: List[str], mode: str):
        for i in range(len(stations_order) - 1):
            # 计算站间时间和距离
            dist = geodesic(
                list(reversed(eval(stations_order[i]["location"]))),
                list(reversed(eval(stations_order[i + 1]["location"]))),
            ).m
            time = dist / (self._metro_speed if mode == "metro" else self._bus_speed) / 60  # min
            self.graph.add_edge(
                stations_order[i]["id"],
                stations_order[i + 1]["id"],
                mode=mode,
                time=time,
                cost=(
                    (dist / 1000) * self._metro_price if mode == "metro" else (dist / 1000) * self._bus_price
                ),
                line=line_name,
            )

    def get_possible_actions(self, current_node):
        """获取当前节点的所有可能动作（下一节点及交通方式）"""
        neighbors = []
        if not self.graph.has_node(current_node):
            return neighbors
        for idx, neighbor in enumerate(self.graph.neighbors(current_node)):
            neighbors.append(
                (
                    neighbor,
                    self.graph.nodes[neighbor]["name"]
                    + "("
                    + self.graph.nodes[neighbor]["line"]
                    + ")",
                    self.graph.nodes[neighbor]["location"],
                    self.graph.get_edge_data(current_node, neighbor)['mode'],
                )
            )

        return neighbors



    def calculate_path_cost(self, path: List[str]) -> float:
        """计算路径总成本（论文公式1）"""
        total = 0
        prev_mode = None
        transfer_count = 0

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = self.graph.get_edge_data(u, v)

            # 时间成本（F1）
            time_cost = edge_data["time"] * self._time_value()
            # 经济成本（F2）
            money_cost = edge_data["cost"]
            # 换乘惩罚（F3）
            if prev_mode and edge_data["mode"] != prev_mode or edge_data["mode"] == "walk":
                transfer_count += 1

            total += 10 * time_cost + 2 * money_cost
            prev_mode = edge_data["mode"]

        # 添加换乘惩罚（论文公式14）
        total += 5000 * transfer_count * self._time_value()
        return total

    def update_real_time_traffic(self, traffic_data):
        """实时交通更新（论文中的动态要素）"""
        # 可以通过手动设置交通信息进行假设
        for road in traffic_data.get("roads", []):
            for u, v, key in self.graph.edges(keys=True):
                edge_data = self.graph[u][v][key]
                if edge_data["mode"] == "bus" and road["name"] in edge_data.get(
                        "line", ""
                ):
                    # 根据拥堵等级调整时间
                    congestion = int(road.get("status", 0))
                    time_multiplier = 1 + congestion * 0.3
                    self.graph[u][v][key]["time"] = (
                            edge_data["base_time"] * time_multiplier
                    )

    def _time_value(self) -> float:
        """单位分钟的时间价值（论文公式3）"""
        # 假设武汉平均月收入7000元，月工作174小时
        return (7000 / 174) * 0.4 / 60.0  # 使用论文中的修正系数

    def visualize_network(self):
        """网络可视化（调试用）"""
        import matplotlib.pyplot as plt

        pos = {
            n: d["location"][::-1] for n, d in self.graph.nodes(data=True)
        }  # 经纬度转坐标
        edge_colors = {
            "bus": "green",
            "metro": "red",
            "car_share": "blue",
            "transfer": "gray",
        }

        plt.figure(figsize=(15, 15))
        nx.draw(self.graph)
        plt.show()

    def save_network(self, filename: str = "wuhan_multimodal_network.gexf"):
        """保存网络数据（供算法使用）"""
        nx.write_gexf(self.graph, filename)
        print(f"网络已保存至 {filename}，包含：")
        print(f"- {self.graph.number_of_nodes()} 个节点")
        print(f"- {self.graph.number_of_edges()} 条边")

    import networkx as nx

    def filter_region_edge(self, start_location, end_location):
        # 假设 start_location 与 end_location 的格式为 (经度, 纬度)
        start_lon, start_lat = start_location
        end_lon, end_lat = end_location

        # 计算矩形区域的边界
        min_lon, max_lon = min(start_lon, end_lon), max(start_lon, end_lon)
        min_lat, max_lat = min(start_lat, end_lat), max(start_lat, end_lat)

        nodes_to_remove = []

        # 遍历图中所有节点
        for node, data in self.graph.nodes(data=True):
            loc_str = data["location"]

            lon_str, lat_str = loc_str.split(',')
            lon, lat = float(lon_str), float(lat_str)

            # 判断节点是否在矩形区域内（以经度和纬度为条件）
            if not (min_lon <= lon <= max_lon and min_lat <= lat <= max_lat):
                nodes_to_remove.append(node)

        # 移除节点以及相关的边
        self.graph.remove_nodes_from(nodes_to_remove)

    def reset(self):  # 重置环境
        start, goal = random.sample(list(self.graph.nodes()), 2)
        self.count = 0
        state = np.array([start, goal])
        self.goal = goal
        return state

    def step(self, action):
        idx, next1, name, location, mode = action

        next_state = np.array([next1, self.goal])
        self.count += 1

        # 1. 步长（路径代价）惩罚
        reward = self.reward(self.current, next1, self.goal)

        goal_done = False
        game_done = False

        if next1 == self.goal:
            goal_done = True
            game_done = True
        if next1 != self.goal:
            goal_done = False
            if self.count >= 50:
                game_done = True
            else:
                game_done = False

        return next_state, reward, goal_done, game_done

    def set_current(self, current):
        self.current = current

    def reward(self, current, next1, goal):
        # 1. 步长（路径代价）惩罚
        step_cost = -self.calculate_path_cost([current, next1])

        # 2. 目标达成奖励：当达到目标时给予正奖励（这里假设 self.beta 为预设目标奖励值）
        goal_reward = 0
        if next1 == goal:
            goal_reward = self.reach_goal

        # 3. 奖励形状（Potential-Based Reward Shaping）
        # 定义潜在函数：Φ(s) = - 距离(s, goal)
        current_location = self.graph.nodes[current]["location"]
        goal_location = self.graph.nodes[goal]["location"]
        next_location = self.graph.nodes[next1]["location"]

        current_potential = -self.get_dist(current_location, goal_location)
        next_potential = -self.get_dist(next_location, goal_location)
        shaping_reward = self.gamma * next_potential - current_potential

        # 综合奖励：步长惩罚 + 奖励形状 + 目标奖励
        reward = step_cost + shaping_reward + goal_reward
        return reward

    def initialize_encoder(self):
        # 从图中获取所有的节点编号
        all_states = list(self.graph.nodes())
        # 如果encoder需要字符串，可以转成字符串：all_states = [str(n) for n in self.graph.nodes()]
        self.encoder.fit(all_states)

    def calculate_transfer_time(self, first_id, second_id):
        """
        计算换乘时间，包括步行时间与等待时间
        :param first_id: 站点id
        :param second_id:
        :return: 换乘分钟
        """
        dist = self.get_dist(
            self.graph.nodes[first_id]["location"],
            self.graph.nodes[second_id]["location"],
        )
        walk_time = dist / self._walk_speed / 60.0
        return walk_time + self._wait_car_time




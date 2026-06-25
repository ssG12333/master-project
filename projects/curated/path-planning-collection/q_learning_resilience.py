import sys
import os
import json
import math
import random
import pickle
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QPainter, QPen, QFont

# ---------------------- 图节点与边的处理 ----------------------
class Graph:
    def __init__(self, json_path):
        self.nodes = {}  # id: (x,y)
        self.edges = {}  # id: list of (neighbor_id, length)
        # 保存 JSON 文件名（不含扩展名）用于存储权重
        self.json_name = os.path.splitext(os.path.basename(json_path))[0]
        self.load_json(json_path)

    def load_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        idx = 0
        for feat in data['features']:
            coords = feat['geometry']['coordinates']
            length = feat['properties']['length']
            for c in coords:
                if tuple(c) not in self.nodes.values():
                    self.nodes[idx] = tuple(c)
                    idx += 1
            id1 = self.find_node_id(coords[0])
            id2 = self.find_node_id(coords[-1])
            self.edges.setdefault(id1, []).append((id2, length))
            self.edges.setdefault(id2, []).append((id1, length))

    def find_node_id(self, coord):
        for nid, c in self.nodes.items():
            if c == tuple(coord):
                return nid
        return None

    def euclidean(self, id1, id2):
        x1, y1 = self.nodes[id1]
        x2, y2 = self.nodes[id2]
        return math.hypot(x1 - x2, y1 - y2)

# ---------------------- Q-learning 与 A* 融合算法 ----------------------
class PathFinder:
    def __init__(self, graph: Graph, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.graph = graph
        self.Q = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def train(self, start, goal, episodes=1000):
        # 使用与 JSON 文件同名的文件夹存储权重
        folder = self.graph.json_name
        os.makedirs(folder, exist_ok=True)
        for ep in range(episodes):
            state = start
            while state != goal:
                if random.random() < self.epsilon:
                    action = random.choice(self.graph.edges[state])[0]
                else:
                    action = self.best_action(state)

                length = next(l for (n, l) in self.graph.edges[state] if n == action)
                goal_reward = 100 if action == goal else 0
                dist_before = self.graph.euclidean(state, goal)
                dist_after = self.graph.euclidean(action, goal)
                distance_reward = dist_before - dist_after
                length_penalty = -length
                reward = 1.0 * goal_reward + 0.5 * distance_reward + 0.2 * length_penalty

                old_q = self.Q.get((state, action), 0)
                next_max = max([self.Q.get((action, a[0]), 0) for a in self.graph.edges[action]] or [0])
                self.Q[(state, action)] = old_q + self.alpha * (reward + self.gamma * next_max - old_q)
                state = action
            print(f"Episode {ep+1}/{episodes} 完成")

        # Change the filename to node-node.pkl
        fname = os.path.join(folder, f'{start}-{goal}.pkl')
        with open(fname, 'wb') as f:
            pickle.dump(self.Q, f)
        print(f"训练完成，权重已保存: {fname}")

    def load_weights(self, folder):
        try:
            # Load all weight files in the folder
            self.Q = {}
            for fname in os.listdir(folder):
                if fname.endswith('.pkl'):
                    with open(os.path.join(folder, fname), 'rb') as f:
                        weight_data = pickle.load(f)
                        self.Q.update(weight_data)
            print("所有权重加载完成")
        except Exception as e:
            print(f"权重加载失败: {e}")

    def best_action(self, state):
        actions = self.graph.edges[state]
        q_vals = [(self.Q.get((state, a[0]), 0), a[0]) for a in actions]
        return max(q_vals)[1]

    def astar(self, start, goal):
        open_set = {start}
        came_from = {}
        g_score = {n: float('inf') for n in self.graph.nodes}
        f_score = {n: float('inf') for n in self.graph.nodes}
        g_score[start] = 0
        f_score[start] = self.graph.euclidean(start, goal)
        while open_set:
            current = min(open_set, key=lambda n: f_score[n])
            if current == goal:
                return self.reconstruct_path(came_from, current)
            open_set.remove(current)
            for neighbor, length in self.graph.edges[current]:
                tentative_g = g_score[current] + length
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.graph.euclidean(neighbor, goal)
                    open_set.add(neighbor)
        return []

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def test(self, start, goal):
        path = [start]
        state = start
        while state != goal:
            action = self.best_action(state)
            path.append(action)
            state = action
            if len(path) > len(self.graph.nodes):
                print("Q策略失败，使用A*算法")
                return self.astar(start, goal)
        return path

# ---------------------- Qt界面 ----------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Q-learning + A* 最短路径")
        self.setGeometry(100, 100, 1000, 800)
        self.graph = None
        self.pathfinder = None
        self.start = None
        self.goal = None
        self.path = None
        self.info = QLabel("点击加载JSON后，在地图上点击两个节点分别作为起点(绿色)和终点(蓝色)，然后训练或测试", self)
        self.info.setGeometry(10, 10, 800, 20)
        self.btnLoad = QPushButton("加载JSON", self); self.btnLoad.setGeometry(10,40,100,30); self.btnLoad.clicked.connect(self.load_json)
        self.btnTrain = QPushButton("训练", self); self.btnTrain.setGeometry(120,40,100,30); self.btnTrain.clicked.connect(self.train)
        self.btnTest  = QPushButton("测试", self); self.btnTest.setGeometry(230,40,100,30); self.btnTest.clicked.connect(self.test)
        self.btnLoadWeights = QPushButton("加载权重", self); self.btnLoadWeights.setGeometry(340,40,100,30); self.btnLoadWeights.clicked.connect(self.load_weights)

    def load_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择JSON文件", "", "JSON Files (*.json)")
        if path:
            self.graph = Graph(path)
            self.pathfinder = PathFinder(self.graph)
            self.scale_params()
            self.update()

    def load_weights(self, folder):
        if folder and self.pathfinder:
            self.pathfinder.load_weights(folder)

    def scale_params(self):
        xs = [c[0] for c in self.graph.nodes.values()]
        ys = [c[1] for c in self.graph.nodes.values()]
        self.minx, self.maxx = min(xs), max(xs)
        self.miny, self.maxy = min(ys), max(ys)
        self.w_scale = (self.width()-50)/(self.maxx-self.minx)
        self.h_scale = (self.height()-100)/(self.maxy-self.miny)

    def train(self):
        if self.start is not None and self.goal is not None:
            self.pathfinder.train(self.start, self.goal)

    def test(self):
        if self.start is not None and self.goal is not None:
            self.path = self.pathfinder.test(self.start, self.goal)
            self.update()

    def mousePressEvent(self, event):
        if not self.graph: return
        x, y = event.x(), event.y()
        min_d, nid = float('inf'), None
        for id, coord in self.graph.nodes.items():
            sx, sy = self.toScreen(coord)
            d = math.hypot(x-sx, y-sy)
            if d<min_d: min_d, nid = d, id
        if event.button()==Qt.LeftButton:
            self.start = nid
        elif event.button()==Qt.RightButton:
            self.goal = nid
        self.update()

    def paintEvent(self, event):
        if not self.graph: return
        qp = QPainter(self)
        pen = QPen(Qt.black, 1)
        qp.setPen(pen)
        for u, edges in self.graph.edges.items():
            for v,_ in edges:
                x1,y1 = self.toScreen(self.graph.nodes[u]); x2,y2 = self.toScreen(self.graph.nodes[v])
                qp.drawLine(x1,y1,x2,y2)
        qp.setFont(QFont('Arial', 8))
        for id, coord in self.graph.nodes.items():
            x,y = self.toScreen(coord)
            qp.drawEllipse(x-3,y-3,6,6)
            qp.drawText(x+5,y-5,str(id))
        if self.start is not None:
            pen.setColor(Qt.green); pen.setWidth(6); qp.setPen(pen)
            x,y = self.toScreen(self.graph.nodes[self.start]); qp.drawPoint(x,y)
        if self.goal is not None:
            pen.setColor(Qt.blue); pen.setWidth(6); qp.setPen(pen)
            x,y = self.toScreen(self.graph.nodes[self.goal]); qp.drawPoint(x,y)
        if self.path:
            pen.setColor(Qt.red); pen.setWidth(4); qp.setPen(pen)
            for i in range(len(self.path)-1):
                x1,y1 = self.toScreen(self.graph.nodes[self.path[i]])
                x2,y2 = self.toScreen(self.graph.nodes[self.path[i+1]])
                qp.drawLine(x1,y1,x2,y2)

    def toScreen(self, coord):
        lon, lat = coord
        x = int((lon-self.minx)*self.w_scale)+20
        y = int((self.maxy-lat)*self.h_scale)+60
        return x, y

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

import math
import heapq

class Node:
    def __init__(self, x, y, theta, direction, g, h, f, node_index, parent_index):
        self.x = x
        self.y = y
        self.theta = theta
        self.direction = direction
        self.g = g
        self.h = h
        self.f = f
        self.node_index = node_index
        self.parent_index = parent_index
        self.route = []
    
    def __lt__(self, other):
        return self.f < other.f

def astar_distance(start, goal, grid):
    rows, cols = len(grid), len(grid[0])
    
    not_cross = 1
    mov = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    open_list = [start]
    closed = []
    
    parent = [[0] * (cols + 1) for _ in range(rows + 1)]
    in_open = [[False] * (cols + 1) for _ in range(rows + 1)]
    in_close = [[False] * (cols + 1) for _ in range(rows + 1)]
    G = [[float('inf')] * (cols + 1) for _ in range(rows + 1)]
    F = [[float('inf')] * (cols + 1) for _ in range(rows + 1)]
    
    in_open[start[0]][start[1]] = True
    G[start[0]][start[1]] = 0
    F[start[0]][start[1]] = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    
    while open_list:
        min_f = float('inf')
        min_node = None
        min_index = -1
        for i, node in enumerate(open_list):
            f_val = F[node[0]][node[1]]
            if f_val < min_f:
                min_f = f_val
                min_node = node
                min_index = i
        
        open_list.pop(min_index)
        in_open[min_node[0]][min_node[1]] = False
        closed.append(min_node)
        in_close[min_node[0]][min_node[1]] = True
        
        if min_node[0] == goal[0] and min_node[1] == goal[1]:
            return G[goal[0]][goal[1]]
        
        for i, (dy, dx) in enumerate(mov):
            temp = (min_node[0] + dy, min_node[1] + dx)
            
            if 0 < temp[0] <= rows and 0 < temp[1] <= cols:
                if grid[temp[0]-1][temp[1]-1] != not_cross and not in_close[temp[0]][temp[1]]:
                    if dy == 0 or dx == 0:
                        if not in_open[temp[0]][temp[1]]:
                            parent[temp[0]][temp[1]] = (min_node[1] - 1) * rows + min_node[0]
                            open_list.append(temp)
                            G[temp[0]][temp[1]] = G[min_node[0]][min_node[1]] + math.sqrt(dy**2 + dx**2)
                            F[temp[0]][temp[1]] = G[temp[0]][temp[1]] + abs(temp[0] - goal[0]) + abs(temp[1] - goal[1])
                            in_open[temp[0]][temp[1]] = True
                        else:
                            gnn = math.sqrt(dy**2 + dx**2) + G[min_node[0]][min_node[1]]
                            if gnn < G[temp[0]][temp[1]]:
                                parent[temp[0]][temp[1]] = (min_node[1] - 1) * rows + min_node[0]
                                G[temp[0]][temp[1]] = gnn
                                F[temp[0]][temp[1]] = G[temp[0]][temp[1]] + abs(temp[0] - goal[0]) + abs(temp[1] - goal[1])
                    else:
                        if grid[min_node[0] + dy - 1][min_node[1] - 1] != not_cross and \
                           grid[min_node[0] - 1][min_node[1] + dx - 1] != not_cross:
                            if not in_open[temp[0]][temp[1]]:
                                parent[temp[0]][temp[1]] = (min_node[1] - 1) * rows + min_node[0]
                                open_list.append(temp)
                                G[temp[0]][temp[1]] = G[min_node[0]][min_node[1]] + math.sqrt(dy**2 + dx**2)
                                F[temp[0]][temp[1]] = G[temp[0]][temp[1]] + abs(temp[0] - goal[0]) + abs(temp[1] - goal[1])
                                in_open[temp[0]][temp[1]] = True
                            else:
                                gnn = math.sqrt(dy**2 + dx**2) + G[min_node[0]][min_node[1]]
                                if gnn < G[temp[0]][temp[1]]:
                                    parent[temp[0]][temp[1]] = (min_node[1] - 1) * rows + min_node[0]
                                    G[temp[0]][temp[1]] = gnn
                                    F[temp[0]][temp[1]] = G[temp[0]][temp[1]] + abs(temp[0] - goal[0]) + abs(temp[1] - goal[1])
    
    return float('inf')
